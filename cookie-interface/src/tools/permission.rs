//! Deciding whether to ask.
//!
//! The rule this is built around: **you should be able to decide without
//! reading a shell command.** "I'm about to remove the old build files, is
//! that okay?" is a question anybody can answer. "Execute: rm -rf ./target"
//! is a question that trains people to say yes to everything, which is worse
//! than not asking at all.
//!
//! The other rule: "you don't need to ask me again for this" is honoured, and
//! bounded. It applies to the task at hand, not to the rest of time, and it
//! never covers the irreversible things — because the one operation you most
//! want to be asked about is the one you will most regret waving through.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::Risk;

/// What to do with a request.
#[derive(Debug, Clone, PartialEq)]
pub enum Decision {
    /// Go ahead.
    Allow,
    /// Ask first, using this sentence.
    Confirm { question: String, risk: Risk },
    /// Refuse, and say why.
    Refuse { reason: String },
}

/// How far a "stop asking" instruction reaches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Suppression {
    /// Ask every time.
    Never,
    /// Stop asking for the rest of this task.
    Task,
    /// Stop asking until the application restarts.
    Session,
}

/// The confirmation policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PermissionPolicy {
    /// Anything at or below this risk happens without asking.
    ///
    /// `Safe` by default: reading, searching and checking need no ceremony,
    /// and requiring it for them is how you end up with a user who confirms
    /// reflexively.
    pub auto_approve_up_to: Risk,
    /// Refuse anything above this outright, whatever the user says.
    /// `Critical` by default — refuse nothing, ask about everything.
    pub refuse_above: Risk,
    /// Critical operations are always confirmed, even when the user has said
    /// to stop asking. This is the safeguard that is not negotiable.
    pub always_confirm_critical: bool,
    /// Tools that are never available on this machine, by name.
    pub blocked: BTreeSet<String>,
    /// Current suppression scope, set by "no need to ask for this task".
    #[serde(skip)]
    pub suppressed: Option<(Suppression, Risk)>,
}

impl Default for PermissionPolicy {
    fn default() -> Self {
        Self {
            auto_approve_up_to: Risk::Safe,
            refuse_above: Risk::Critical,
            always_confirm_critical: true,
            blocked: BTreeSet::new(),
            suppressed: None,
        }
    }
}

impl PermissionPolicy {
    /// What to do about an operation of this risk.
    pub fn decide(&self, risk: Risk, description: &str) -> Decision {
        if risk > self.refuse_above {
            return Decision::Refuse {
                reason: format!("I'm not allowed to do that on this machine. {description}"),
            };
        }
        if risk <= self.auto_approve_up_to {
            return Decision::Allow;
        }
        if self.is_suppressed(risk) {
            return Decision::Allow;
        }
        Decision::Confirm {
            question: question_for(risk, description),
            risk,
        }
    }

    fn is_suppressed(&self, risk: Risk) -> bool {
        if risk == Risk::Critical && self.always_confirm_critical {
            // The point of the safeguard: "stop asking" never reaches here.
            return false;
        }
        match self.suppressed {
            Some((_, ceiling)) => risk <= ceiling,
            None => false,
        }
    }

    /// Honour "you don't need to ask me for the rest of this".
    ///
    /// Deliberately capped at `Dangerous`: a blanket yes should cover the
    /// tedious middle, not the irreversible end.
    pub fn suppress(&mut self, scope: Suppression) {
        let ceiling = Risk::Dangerous;
        self.suppressed = match scope {
            Suppression::Never => None,
            other => Some((other, ceiling)),
        };
    }

    /// Called when a task ends, so a task-scoped suppression expires.
    pub fn end_task(&mut self) {
        if matches!(self.suppressed, Some((Suppression::Task, _))) {
            self.suppressed = None;
        }
    }

    /// Whether confirmations are currently being skipped.
    pub fn suppression(&self) -> Option<Suppression> {
        self.suppressed.map(|(scope, _)| scope)
    }
}

/// Turn a description of the consequence into a question.
///
/// The phrasing escalates with the risk, because "is that okay?" is the wrong
/// register for something irreversible.
fn question_for(risk: Risk, description: &str) -> String {
    let description = description.trim().trim_end_matches('.');
    match risk {
        Risk::Critical => format!("{description}. That can't be undone. Are you sure?"),
        Risk::Dangerous => format!("{description}. Can you confirm?"),
        _ => format!("{description}. Is that okay?"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> PermissionPolicy {
        PermissionPolicy::default()
    }

    #[test]
    fn reading_things_does_not_interrupt_you() {
        assert_eq!(
            policy().decide(Risk::Safe, "I'm about to read the file"),
            Decision::Allow
        );
    }

    #[test]
    fn writing_asks_and_deleting_asks_more_firmly() {
        match policy().decide(Risk::Normal, "I'm about to update the config file") {
            Decision::Confirm { question, .. } => assert!(question.ends_with("Is that okay?")),
            other => panic!("{other:?}"),
        }
        match policy().decide(Risk::Dangerous, "I'm about to delete the old build files") {
            Decision::Confirm { question, .. } => assert!(question.ends_with("Can you confirm?")),
            other => panic!("{other:?}"),
        }
        match policy().decide(Risk::Critical, "I'm about to force push over the branch") {
            Decision::Confirm { question, .. } => assert!(question.contains("can't be undone")),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn questions_describe_the_consequence_not_the_command() {
        // The whole point: somebody who does not read shell can still decide.
        // (Checking for "rm" as a substring would match "confirm", which is
        // a good reminder that naive string assertions are how you end up
        // trusting a test that is measuring the wrong thing.)
        let decision = policy().decide(Risk::Dangerous, "I'm about to remove the old build files");
        match decision {
            Decision::Confirm { question, .. } => {
                assert!(question.starts_with("I'm about to remove the old build files"));
                for shell in ["rm -rf", "sudo", "--force", "$", "&&"] {
                    assert!(!question.contains(shell), "{question} leaked {shell}");
                }
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn stop_asking_covers_the_tedious_middle() {
        let mut policy = policy();
        policy.suppress(Suppression::Task);
        assert_eq!(
            policy.decide(Risk::Normal, "writing a file"),
            Decision::Allow
        );
        assert_eq!(
            policy.decide(Risk::Dangerous, "deleting a file"),
            Decision::Allow
        );
    }

    #[test]
    fn stop_asking_never_reaches_the_irreversible() {
        let mut policy = policy();
        policy.suppress(Suppression::Session);
        assert!(matches!(
            policy.decide(Risk::Critical, "force pushing"),
            Decision::Confirm { .. }
        ));
    }

    #[test]
    fn a_task_scoped_suppression_expires_with_the_task() {
        let mut policy = policy();
        policy.suppress(Suppression::Task);
        assert_eq!(policy.decide(Risk::Normal, "writing"), Decision::Allow);
        policy.end_task();
        assert!(matches!(
            policy.decide(Risk::Normal, "writing"),
            Decision::Confirm { .. }
        ));

        // A session-scoped one does not.
        policy.suppress(Suppression::Session);
        policy.end_task();
        assert_eq!(policy.decide(Risk::Normal, "writing"), Decision::Allow);
    }

    #[test]
    fn a_lowered_ceiling_refuses_rather_than_asks() {
        let mut policy = policy();
        policy.refuse_above = Risk::Normal;
        match policy.decide(Risk::Dangerous, "I'm about to delete the project") {
            Decision::Refuse { reason } => assert!(reason.contains("not allowed")),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_raised_floor_stops_asking_about_ordinary_edits() {
        let mut policy = policy();
        policy.auto_approve_up_to = Risk::Normal;
        assert_eq!(
            policy.decide(Risk::Normal, "writing a file"),
            Decision::Allow
        );
        assert!(matches!(
            policy.decide(Risk::Dangerous, "deleting a file"),
            Decision::Confirm { .. }
        ));
    }
}
