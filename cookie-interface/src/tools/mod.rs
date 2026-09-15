//! Doing things on this machine, when the backend asks.
//!
//! The backend has the ideas; this side has the filesystem, the shell, the
//! applications and the person sitting in front of them. That split is the
//! security boundary of the whole system: the backend sends a *structured
//! request* — `shell.run` with arguments — never a command string to
//! evaluate, and everything it asks for passes through here first.
//!
//! Three things happen to every request:
//!
//! 1. **The tool must exist and the arguments must fit.** A hallucinated tool
//!    or a hallucinated argument is refused with a sentence the model can act
//!    on, not a panic.
//! 2. **The risk decides whether you are asked.** Reading a file is not
//!    deleting one. See [`permission`].
//! 3. **The result is structured.** Exit codes, output, what changed — the
//!    backend's validator judges against this, so "it worked" has to be
//!    something other than an opinion.
//!
//! What this module deliberately does not do is decide *what* to run. There
//! is no planning here and no model; it executes one named operation and
//! reports what happened.

mod filesystem;
pub mod permission;
mod shell;
mod system;

use std::collections::BTreeMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::Result;
use crate::util::BoxFuture;

pub use permission::{Decision, PermissionPolicy, Suppression};

/// How much damage a tool can do if the model is wrong about wanting it.
///
/// Ordered, so a policy can say "anything up to `Normal` without asking".
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Risk {
    /// Read-only: listing, searching, reading, checking whether something
    /// exists.
    Safe,
    /// Recoverable writes: editing a file, committing, opening an application.
    Normal,
    /// Deletes, installs, anything touching credentials.
    Dangerous,
    /// Irreversible: force pushes, disk operations.
    Critical,
}

impl Risk {
    pub fn as_str(self) -> &'static str {
        match self {
            Risk::Safe => "safe",
            Risk::Normal => "normal",
            Risk::Dangerous => "dangerous",
            Risk::Critical => "critical",
        }
    }
}

/// A request from the backend.
#[derive(Debug, Clone, Deserialize)]
pub struct ToolRequest {
    pub id: String,
    #[serde(rename = "tool")]
    pub name: String,
    #[serde(default = "default_version")]
    pub version: u32,
    #[serde(default)]
    pub arguments: Value,
    /// The backend's view of the risk. Advisory: this side uses its own
    /// classification, because this is the machine being acted upon and a
    /// backend that has been talked into something must not be able to talk
    /// its way past the confirmation policy too.
    #[serde(default)]
    pub risk: Option<Risk>,
}

fn default_version() -> u32 {
    1
}

/// What happened, in the shape the backend's validator reads.
#[derive(Debug, Clone, Serialize)]
pub struct ToolOutcome {
    pub id: String,
    pub ok: bool,
    /// One or two lines for the model, and possibly for speech.
    pub summary: String,
    /// What *shows* it worked: an exit code, a path, a count. The validator
    /// judges this rather than the summary, so it must be factual.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub evidence: String,
    /// Structured output for tools feeding other tools.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl ToolOutcome {
    pub fn ok(id: &str, summary: impl Into<String>, evidence: impl Into<String>) -> Self {
        Self {
            id: id.to_string(),
            ok: true,
            summary: summary.into(),
            evidence: evidence.into(),
            data: None,
            error: None,
        }
    }

    pub fn failed(id: &str, error: impl Into<String>) -> Self {
        let error = error.into();
        Self {
            id: id.to_string(),
            ok: false,
            summary: error.clone(),
            evidence: error.clone(),
            data: None,
            error: Some(error),
        }
    }

    pub fn with_data(mut self, data: Value) -> Self {
        self.data = Some(data);
        self
    }
}

/// One thing this machine can do.
pub trait LocalTool: Send + Sync + 'static {
    /// Must match the backend's catalogue exactly.
    fn name(&self) -> &'static str;

    /// This side's classification, which is the one that counts.
    fn risk(&self, arguments: &Value) -> Risk;

    /// What to say when asking permission.
    ///
    /// Plain English describing the *consequence*, never the command. "I'm
    /// about to remove the old build files" — not "execute rm -rf target".
    /// Somebody who does not read shell should still be able to decide.
    fn describe(&self, arguments: &Value) -> String;

    /// Do it.
    fn run<'a>(&'a self, id: &'a str, arguments: Value) -> BoxFuture<'a, ToolOutcome>;
}

/// Everything this machine offers.
pub struct ToolHost {
    tools: BTreeMap<&'static str, Arc<dyn LocalTool>>,
    policy: PermissionPolicy,
}

impl std::fmt::Debug for ToolHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolHost")
            .field("tools", &self.tools.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl ToolHost {
    /// The standard set for this platform.
    pub fn new(policy: PermissionPolicy) -> Self {
        let mut host = Self {
            tools: BTreeMap::new(),
            policy,
        };
        for tool in filesystem::tools() {
            host.register(tool);
        }
        for tool in shell::tools() {
            host.register(tool);
        }
        for tool in system::tools() {
            host.register(tool);
        }
        host
    }

    /// A host with no tools, for tests and for anyone who wants to opt out.
    pub fn empty(policy: PermissionPolicy) -> Self {
        Self {
            tools: BTreeMap::new(),
            policy,
        }
    }

    pub fn register(&mut self, tool: Arc<dyn LocalTool>) {
        self.tools.insert(tool.name(), tool);
    }

    /// Names to advertise to the backend. Anything not here is never asked
    /// for, so a request needing it gets planned differently rather than
    /// failing halfway through.
    pub fn advertised(&self) -> Vec<String> {
        self.tools.keys().map(|name| name.to_string()).collect()
    }

    pub fn policy(&self) -> &PermissionPolicy {
        &self.policy
    }

    pub fn policy_mut(&mut self) -> &mut PermissionPolicy {
        &mut self.policy
    }

    /// Decide what to do about a request without doing it.
    pub fn consider(
        &self,
        request: &ToolRequest,
    ) -> std::result::Result<Decision, Box<ToolOutcome>> {
        let Some(tool) = self.tools.get(request.name.as_str()) else {
            return Err(Box::new(ToolOutcome::failed(
                &request.id,
                format!("this machine has no tool called {}", request.name),
            )));
        };
        let risk = tool.risk(&request.arguments);
        Ok(self.policy.decide(risk, &tool.describe(&request.arguments)))
    }

    /// Run a request that has already been authorised.
    pub async fn run(&self, request: &ToolRequest) -> ToolOutcome {
        let Some(tool) = self.tools.get(request.name.as_str()) else {
            return ToolOutcome::failed(
                &request.id,
                format!("this machine has no tool called {}", request.name),
            );
        };
        tool.run(&request.id, request.arguments.clone()).await
    }

    /// The risk this side assigns, for logging and for the event stream.
    pub fn risk_of(&self, request: &ToolRequest) -> Option<Risk> {
        self.tools
            .get(request.name.as_str())
            .map(|tool| tool.risk(&request.arguments))
    }
}

/// Helpers shared by the tool implementations.
pub(crate) fn string_arg(arguments: &Value, key: &str) -> Option<String> {
    arguments
        .get(key)
        .and_then(Value::as_str)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

pub(crate) fn require_arg(arguments: &Value, key: &str) -> std::result::Result<String, String> {
    string_arg(arguments, key).ok_or_else(|| format!("that needs a {key}"))
}

/// Expand a leading `~` and make the path absolute.
///
/// Models produce `~/projects` constantly, and nothing in the standard
/// library expands it.
pub(crate) fn resolve_path(raw: &str) -> std::path::PathBuf {
    let raw = raw.trim();
    if let Some(rest) = raw.strip_prefix("~/") {
        if let Some(home) = home_dir() {
            return home.join(rest);
        }
    }
    if raw == "~" {
        if let Some(home) = home_dir() {
            return home;
        }
    }
    std::path::PathBuf::from(raw)
}

pub(crate) fn home_dir() -> Option<std::path::PathBuf> {
    directories::UserDirs::new().map(|dirs| dirs.home_dir().to_path_buf())
}

/// A convenience for tools that are simple async functions.
pub(crate) struct SimpleTool {
    pub name: &'static str,
    pub risk: fn(&Value) -> Risk,
    pub describe: fn(&Value) -> String,
    #[allow(clippy::type_complexity)]
    pub run: fn(String, Value) -> BoxFuture<'static, ToolOutcome>,
}

impl LocalTool for SimpleTool {
    fn name(&self) -> &'static str {
        self.name
    }

    fn risk(&self, arguments: &Value) -> Risk {
        (self.risk)(arguments)
    }

    fn describe(&self, arguments: &Value) -> String {
        (self.describe)(arguments)
    }

    fn run<'a>(&'a self, id: &'a str, arguments: Value) -> BoxFuture<'a, ToolOutcome> {
        (self.run)(id.to_string(), arguments)
    }
}

/// Never used directly; keeps the `Result` import honest for tool authors.
#[allow(dead_code)]
fn _result_anchor() -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn risks_are_ordered_so_a_policy_can_use_a_threshold() {
        assert!(Risk::Safe < Risk::Normal);
        assert!(Risk::Normal < Risk::Dangerous);
        assert!(Risk::Dangerous < Risk::Critical);
    }

    #[test]
    fn unknown_tools_are_refused_with_something_a_model_can_act_on() {
        let host = ToolHost::new(PermissionPolicy::default());
        let request = ToolRequest {
            id: "c1".into(),
            name: "filesystem.teleport".into(),
            version: 1,
            arguments: json!({}),
            risk: Some(Risk::Safe),
        };
        let outcome = host.consider(&request).unwrap_err();
        assert!(!outcome.ok);
        assert!(outcome.summary.contains("no tool called"));
    }

    #[test]
    fn the_backends_risk_claim_does_not_override_ours() {
        let host = ToolHost::new(PermissionPolicy::default());
        let request = ToolRequest {
            id: "c2".into(),
            name: "filesystem.delete".into(),
            version: 1,
            arguments: json!({"path": "/tmp/whatever"}),
            // The backend says this is harmless. It is not.
            risk: Some(Risk::Safe),
        };
        assert_eq!(host.risk_of(&request), Some(Risk::Dangerous));
        assert!(matches!(
            host.consider(&request).unwrap(),
            Decision::Confirm { .. }
        ));
    }

    #[test]
    fn tilde_paths_are_expanded() {
        let resolved = resolve_path("~/projects/thing");
        assert!(!resolved.to_string_lossy().starts_with('~'));
        assert_eq!(
            resolve_path("/absolute"),
            std::path::PathBuf::from("/absolute")
        );
    }

    #[test]
    fn advertised_names_match_the_backend_catalogue() {
        let host = ToolHost::new(PermissionPolicy::default());
        let names = host.advertised();
        for expected in [
            "filesystem.search",
            "filesystem.read",
            "filesystem.write",
            "filesystem.delete",
            "shell.which",
            "shell.run",
            "app.open",
            "browser.open",
        ] {
            assert!(names.iter().any(|n| n == expected), "missing {expected}");
        }
    }
}
