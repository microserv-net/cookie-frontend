//! Keeping track of what Cookie is working on.
//!
//! The interface does not *run* tasks — the backend does. But it is the only
//! part of the system that knows a human is standing there waiting, so it
//! owns three things:
//!
//! 1. a live picture of what is in flight, so "what are you working on?" has
//!    an answer and the orb can show that something is happening,
//! 2. the distinction between *stop talking* and *stop working*, which are
//!    different words, different intents and different consequences,
//! 3. the **priority hint** attached to every turn, which is what makes
//!    cooperative scheduling possible on a machine that can only really run
//!    one heavy model at a time.
//!
//! ## Why the interface labels priority
//!
//! The backend cannot tell from the text alone whether you are adding to the
//! job it is already doing or interrupting with something small. The
//! interface can: it knows whether a heavy task is running, how long you have
//! been waiting, and that you just spoke a short sentence rather than typing
//! a specification. So each turn carries a hint:
//!
//! | Situation | Priority | `preempt` |
//! |---|---|---|
//! | nothing running | `normal` | `false` |
//! | heavy task running, short new utterance | `interactive` | `true` |
//! | heavy task running, long new utterance | `normal` | `false` |
//! | explicitly asked to wait | `background` | `false` |
//!
//! `preempt: true` does **not** mean "abandon the task". It means "suspend it
//! at your next natural boundary, answer me, then carry on" — the boundaries
//! being the points where the backend is already between things: after a
//! model call returns, before a model is swapped in, while a tool is running,
//! between plan steps. A backend that ignores the hint still works; it is
//! just less pleasant on constrained hardware. The wire format is in
//! `docs/backend.md`.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// How urgent a request is relative to work already running.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum Priority {
    /// A person is waiting for this right now.
    Interactive,
    /// The normal case.
    #[default]
    Normal,
    /// Do it when nothing else needs the machine.
    Background,
}

impl Priority {
    pub fn as_str(self) -> &'static str {
        match self {
            Priority::Interactive => "interactive",
            Priority::Normal => "normal",
            Priority::Background => "background",
        }
    }
}

/// Where a task has got to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TaskState {
    Queued,
    Running,
    /// Yielded at a checkpoint so something more urgent could run. This is
    /// the state that makes a one-model-at-a-time machine usable.
    Suspended,
    Completed,
    Failed,
    Cancelled,
}

impl TaskState {
    /// Still occupying resources, or about to.
    pub fn is_active(self) -> bool {
        matches!(
            self,
            TaskState::Queued | TaskState::Running | TaskState::Suspended
        )
    }

    pub fn as_str(self) -> &'static str {
        match self {
            TaskState::Queued => "queued",
            TaskState::Running => "running",
            TaskState::Suspended => "suspended",
            TaskState::Completed => "completed",
            TaskState::Failed => "failed",
            TaskState::Cancelled => "cancelled",
        }
    }
}

/// How much of the machine a task expects to need.
///
/// Reported by the backend, because only it knows which model it is about to
/// load. The interface uses it for one decision: whether a new utterance
/// should be marked `interactive` and ask for preemption.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum Weight {
    /// A router-sized model, a tool call, a lookup.
    Light,
    #[default]
    Normal,
    /// A large model, a long plan, a full test run.
    Heavy,
}

/// One unit of work the backend is doing for us.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    /// Short human-readable description: "fixing the failing tests".
    pub title: String,
    pub state: TaskState,
    #[serde(default)]
    pub weight: Weight,
    #[serde(default)]
    pub priority: Priority,
    /// Most recent human-readable progress line, for the activity view and
    /// for answering "what are you doing?" out loud.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// Seconds since the task appeared.
    pub age_seconds: u64,
}

struct Entry {
    task: Task,
    started: Instant,
    updated: Instant,
}

/// Live view of backend work.
///
/// Cheap to read from any thread; the lock is only ever held for a field
/// update or a short scan, never across an `await`.
#[derive(Debug)]
pub struct TaskRegistry {
    inner: Mutex<BTreeMap<String, Entry>>,
}

impl std::fmt::Debug for Entry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Entry").field("task", &self.task).finish()
    }
}

impl Default for TaskRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskRegistry {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(BTreeMap::new()),
        }
    }

    /// Record or update a task. Returns the task as it now stands.
    pub fn upsert(
        &self,
        id: &str,
        title: Option<String>,
        state: TaskState,
        weight: Option<Weight>,
        detail: Option<String>,
    ) -> Task {
        let mut guard = self.inner.lock().expect("task registry");
        let now = Instant::now();
        let entry = guard.entry(id.to_string()).or_insert_with(|| Entry {
            task: Task {
                id: id.to_string(),
                title: title.clone().unwrap_or_else(|| "working".into()),
                state,
                weight: weight.unwrap_or_default(),
                priority: Priority::Normal,
                detail: detail.clone(),
                age_seconds: 0,
            },
            started: now,
            updated: now,
        });
        if let Some(title) = title {
            entry.task.title = title;
        }
        if let Some(weight) = weight {
            entry.task.weight = weight;
        }
        if detail.is_some() {
            entry.task.detail = detail;
        }
        entry.task.state = state;
        entry.task.age_seconds = entry.started.elapsed().as_secs();
        entry.updated = now;
        entry.task.clone()
    }

    /// Everything still in flight, oldest first.
    pub fn active(&self) -> Vec<Task> {
        let guard = self.inner.lock().expect("task registry");
        let mut tasks: Vec<Task> = guard
            .values()
            .filter(|e| e.task.state.is_active())
            .map(|e| {
                let mut task = e.task.clone();
                task.age_seconds = e.started.elapsed().as_secs();
                task
            })
            .collect();
        tasks.sort_by_key(|t| std::cmp::Reverse(t.age_seconds));
        tasks
    }

    /// Ids of everything still in flight — what a cancel applies to.
    pub fn active_ids(&self) -> Vec<String> {
        self.active().into_iter().map(|t| t.id).collect()
    }

    /// Whether a heavy task is currently occupying the machine.
    pub fn heavy_in_flight(&self) -> bool {
        self.active()
            .iter()
            .any(|t| t.weight == Weight::Heavy && t.state != TaskState::Suspended)
    }

    /// Drop finished tasks that nobody is going to ask about again.
    pub fn prune(&self, keep_for: Duration) {
        let mut guard = self.inner.lock().expect("task registry");
        guard.retain(|_, e| e.task.state.is_active() || e.updated.elapsed() < keep_for);
    }

    /// A sentence describing the current workload, for speaking aloud.
    pub fn spoken_summary(&self) -> String {
        let active = self.active();
        match active.len() {
            0 => "Nothing at the moment.".to_string(),
            1 => {
                let task = &active[0];
                let detail = task
                    .detail
                    .as_deref()
                    .map(|d| format!(" Right now: {d}"))
                    .unwrap_or_default();
                let suspended = if task.state == TaskState::Suspended {
                    " It's paused while I deal with this."
                } else {
                    ""
                };
                format!("I'm {}.{detail}{suspended}", task.title)
            }
            n => {
                let titles: Vec<&str> = active.iter().take(3).map(|t| t.title.as_str()).collect();
                format!("{n} things: {}.", titles.join(", "))
            }
        }
    }

    /// The priority a new utterance should carry, and whether to ask the
    /// backend to yield for it.
    ///
    /// The rule is deliberately simple and explainable: a short sentence
    /// spoken while something heavy is running is an interruption from a
    /// human, and humans are more expensive to keep waiting than models are.
    pub fn classify(&self, utterance: &str) -> (Priority, bool) {
        let words = utterance.split_whitespace().count();
        if !self.heavy_in_flight() && self.active().is_empty() {
            return (Priority::Normal, false);
        }
        if words <= 12 {
            (Priority::Interactive, true)
        } else {
            // A long request is a new job, not an aside; queue it properly
            // rather than thrashing the machine.
            (Priority::Normal, false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tasks_are_tracked_until_they_finish() {
        let registry = TaskRegistry::new();
        registry.upsert(
            "t1",
            Some("fixing the tests".into()),
            TaskState::Running,
            Some(Weight::Heavy),
            None,
        );
        assert_eq!(registry.active().len(), 1);
        assert!(registry.heavy_in_flight());
        registry.upsert("t1", None, TaskState::Completed, None, None);
        assert!(registry.active().is_empty());
        assert!(!registry.heavy_in_flight());
    }

    #[test]
    fn updates_do_not_lose_the_title() {
        let registry = TaskRegistry::new();
        registry.upsert(
            "t1",
            Some("building the project".into()),
            TaskState::Running,
            None,
            None,
        );
        let task = registry.upsert(
            "t1",
            None,
            TaskState::Running,
            None,
            Some("running cargo test".into()),
        );
        assert_eq!(task.title, "building the project");
        assert_eq!(task.detail.as_deref(), Some("running cargo test"));
    }

    #[test]
    fn a_suspended_heavy_task_frees_the_machine() {
        let registry = TaskRegistry::new();
        registry.upsert("t1", None, TaskState::Running, Some(Weight::Heavy), None);
        assert!(registry.heavy_in_flight());
        registry.upsert("t1", None, TaskState::Suspended, None, None);
        assert!(!registry.heavy_in_flight());
        // Still active, though: it has not been abandoned.
        assert_eq!(registry.active().len(), 1);
    }

    #[test]
    fn short_asides_during_heavy_work_ask_for_preemption() {
        let registry = TaskRegistry::new();
        registry.upsert("t1", None, TaskState::Running, Some(Weight::Heavy), None);
        let (priority, preempt) = registry.classify("what time is it");
        assert_eq!(priority, Priority::Interactive);
        assert!(preempt);
    }

    #[test]
    fn long_requests_queue_instead_of_preempting() {
        let registry = TaskRegistry::new();
        registry.upsert("t1", None, TaskState::Running, Some(Weight::Heavy), None);
        let (priority, preempt) = registry.classify(
            "go through the authentication module and rewrite the token refresh logic \
             so that it retries properly when the network drops",
        );
        assert_eq!(priority, Priority::Normal);
        assert!(!preempt);
    }

    #[test]
    fn an_idle_machine_never_asks_for_preemption() {
        let registry = TaskRegistry::new();
        assert_eq!(registry.classify("hello"), (Priority::Normal, false));
    }

    #[test]
    fn the_spoken_summary_reads_like_speech() {
        let registry = TaskRegistry::new();
        assert_eq!(registry.spoken_summary(), "Nothing at the moment.");
        registry.upsert(
            "t1",
            Some("fixing the failing tests".into()),
            TaskState::Running,
            None,
            Some("running the test suite".into()),
        );
        let summary = registry.spoken_summary();
        assert!(
            summary.starts_with("I'm fixing the failing tests."),
            "{summary}"
        );
        assert!(summary.contains("running the test suite"));
    }

    #[test]
    fn finished_tasks_are_pruned_eventually() {
        let registry = TaskRegistry::new();
        registry.upsert("t1", None, TaskState::Completed, None, None);
        registry.prune(Duration::from_secs(0));
        assert!(registry.active().is_empty());
        assert_eq!(registry.spoken_summary(), "Nothing at the moment.");
    }
}
