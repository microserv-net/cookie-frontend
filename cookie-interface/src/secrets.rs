//! Secrets.
//!
//! "Cookie, can you keep a secret?" has to end with the secret somewhere the
//! operating system protects, not in a configuration file — and not in a file
//! this project encrypts itself, because that would mean inventing a
//! key-management story that macOS, Windows and Linux all already have.
//!
//! | Platform | Where it goes |
//! |---|---|
//! | macOS | Keychain |
//! | Windows | Credential Manager |
//! | Linux | Secret Service (GNOME Keyring, KWallet) |
//!
//! ## The model never sees the secret
//!
//! This is the part worth defending. The backend and its models refer to
//! secrets *symbolically*:
//!
//! ```text
//! secret://github/personal-token
//! ```
//!
//! and the value is substituted here, on this machine, at the moment a tool
//! actually needs it. A model cannot leak what it was never given; a
//! transcript cannot contain it; an event stream cannot carry it. What the
//! backend gets to know is that a secret with that name exists.
//!
//! What the backend can do is *ask* for one to be stored — at which point the
//! frontend prompts you, and you type it. It never travels over the wire in
//! the other direction either.

use std::collections::BTreeSet;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// The service name secrets are filed under in the platform store.
const SERVICE: &str = "dev.Cookie.cookie-interface";

/// A reference to a secret, as it appears in tool arguments and plans.
///
/// `secret://github/personal-token` — a scope and a name, so that "the GitHub
/// token" and "the Postgres password" are obviously different things and can
/// be revoked independently.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SecretRef {
    pub scope: String,
    pub name: String,
}

impl SecretRef {
    pub fn new(scope: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            scope: into_component(scope.into()),
            name: into_component(name.into()),
        }
    }

    /// Parse `secret://scope/name`. Anything else is not a reference.
    pub fn parse(raw: &str) -> Option<Self> {
        let rest = raw.trim().strip_prefix("secret://")?;
        let (scope, name) = rest.split_once('/')?;
        let (scope, name) = (into_component(scope.into()), into_component(name.into()));
        if scope.is_empty() || name.is_empty() {
            return None;
        }
        Some(Self { scope, name })
    }

    /// The key used in the platform store.
    pub fn key(&self) -> String {
        format!("{}/{}", self.scope, self.name)
    }
}

impl fmt::Display for SecretRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "secret://{}/{}", self.scope, self.name)
    }
}

/// Lowercase, and nothing that would make a key ambiguous.
fn into_component(raw: String) -> String {
    raw.trim()
        .chars()
        .filter(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | '.'))
        .flat_map(char::to_lowercase)
        .collect()
}

/// Reads and writes the platform credential store.
///
/// The index of *names* is kept separately, in ordinary application data,
/// because no credential store offers a reliable way to enumerate its own
/// entries. Names are not secret; the values they point at are.
#[derive(Debug)]
pub struct SecretStore {
    index_path: std::path::PathBuf,
    known: BTreeSet<String>,
}

impl SecretStore {
    pub fn open(paths: &crate::paths::Paths) -> Self {
        let index_path = paths.data_dir().join("secrets.json");
        let known = std::fs::read_to_string(&index_path)
            .ok()
            .and_then(|text| serde_json::from_str::<BTreeSet<String>>(&text).ok())
            .unwrap_or_default();
        Self { index_path, known }
    }

    fn save_index(&self) -> Result<()> {
        if let Some(parent) = self.index_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
        }
        let text = serde_json::to_string_pretty(&self.known)?;
        std::fs::write(&self.index_path, text).map_err(|e| Error::io(&self.index_path, e))
    }

    fn entry(reference: &SecretRef) -> Result<keyring::Entry> {
        keyring::Entry::new(SERVICE, &reference.key()).map_err(|e| {
            Error::Other(format!(
                "this machine's credential store would not open: {e}"
            ))
        })
    }

    /// Store a secret, replacing any previous value.
    pub fn store(&mut self, reference: &SecretRef, secret: &str) -> Result<()> {
        if secret.is_empty() {
            return Err(Error::BadRequest("an empty secret is not a secret".into()));
        }
        Self::entry(reference)?
            .set_password(secret)
            .map_err(|e| Error::Other(format!("could not save {reference}: {e}")))?;
        self.known.insert(reference.key());
        self.save_index()
    }

    /// Retrieve a secret. Callers should hold the value as briefly as
    /// possible and never put it in an event, a log or a transcript.
    pub fn get(&self, reference: &SecretRef) -> Result<String> {
        match Self::entry(reference)?.get_password() {
            Ok(secret) => Ok(secret),
            Err(keyring::Error::NoEntry) => Err(Error::BadRequest(format!(
                "I don't have {reference}. Ask me to remember it first."
            ))),
            Err(e) => Err(Error::Other(format!("could not read {reference}: {e}"))),
        }
    }

    pub fn contains(&self, reference: &SecretRef) -> bool {
        self.get(reference).is_ok()
    }

    /// Forget a secret. Removing one that is not there is not an error —
    /// the outcome the caller wanted is the outcome they get.
    pub fn forget(&mut self, reference: &SecretRef) -> Result<bool> {
        let existed = match Self::entry(reference)?.delete_credential() {
            Ok(()) => true,
            Err(keyring::Error::NoEntry) => false,
            Err(e) => return Err(Error::Other(format!("could not forget {reference}: {e}"))),
        };
        self.known.remove(&reference.key());
        self.save_index()?;
        Ok(existed)
    }

    /// The names of everything stored. Never the values.
    pub fn list(&self) -> Vec<SecretRef> {
        self.known
            .iter()
            .filter_map(|key| key.split_once('/'))
            .map(|(scope, name)| SecretRef::new(scope, name))
            .collect()
    }
}

/// Replace every `secret://…` reference in a string with its value.
///
/// Used at the last possible moment, on this machine, when a tool is about to
/// run. The substituted string is not logged, not echoed into an event, and
/// not returned to the backend — a tool that needs a token gets it, and
/// nothing else does.
///
/// Missing secrets are reported rather than silently left as text: a request
/// that ran with the literal string "secret://github/personal-token" as a
/// password would fail in a much more confusing way.
pub fn substitute(store: &SecretStore, text: &str) -> Result<String> {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;

    while let Some(start) = rest.find("secret://") {
        out.push_str(&rest[..start]);
        let tail = &rest[start..];
        let end = tail
            .char_indices()
            .find(|(_, c)| c.is_whitespace() || matches!(c, '"' | '\'' | ',' | ')' | '}'))
            .map(|(index, _)| index)
            .unwrap_or(tail.len());
        let (token, remainder) = tail.split_at(end);

        match SecretRef::parse(token) {
            Some(reference) => out.push_str(&store.get(&reference)?),
            // Not a reference after all; leave it exactly as it was.
            None => out.push_str(token),
        }
        rest = remainder;
    }
    out.push_str(rest);
    Ok(out)
}

/// Whether a string mentions a secret at all. Cheap enough to call on every
/// tool argument.
pub fn mentions_secret(text: &str) -> bool {
    text.contains("secret://")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn references_parse_and_round_trip() {
        let reference = SecretRef::parse("secret://github/personal-token").unwrap();
        assert_eq!(reference.scope, "github");
        assert_eq!(reference.name, "personal-token");
        assert_eq!(reference.to_string(), "secret://github/personal-token");
        assert_eq!(reference.key(), "github/personal-token");
    }

    #[test]
    fn anything_that_is_not_a_reference_is_not_treated_as_one() {
        assert!(SecretRef::parse("github/personal-token").is_none());
        assert!(SecretRef::parse("secret://github").is_none());
        assert!(SecretRef::parse("secret:///token").is_none());
        assert!(SecretRef::parse("https://github.com/x/y").is_none());
    }

    #[test]
    fn components_are_normalised_so_one_secret_has_one_key() {
        let a = SecretRef::new("GitHub", "Personal Token");
        let b = SecretRef::parse("secret://github/personaltoken").unwrap();
        assert_eq!(a.key(), b.key());
    }

    #[test]
    fn substitution_leaves_ordinary_text_alone() {
        let dir = tempfile::tempdir().unwrap();
        let store = SecretStore::open(&crate::paths::Paths::rooted(dir.path()));
        let text = "git clone https://github.com/microserv-net/cookie-frontend";
        assert_eq!(substitute(&store, text).unwrap(), text);
        assert!(!mentions_secret(text));
    }

    #[test]
    fn a_missing_secret_is_reported_rather_than_left_as_text() {
        // Running a command with the literal string
        // "secret://github/personal-token" as a password fails in a far more
        // confusing way than saying it is not there.
        let dir = tempfile::tempdir().unwrap();
        let store = SecretStore::open(&crate::paths::Paths::rooted(dir.path()));
        let error = substitute(&store, "token: secret://github/personal-token").unwrap_err();
        assert!(error.to_string().contains("remember it first"), "{error}");
    }

    #[test]
    fn the_index_holds_names_and_never_values() {
        // The index is ordinary application data, so this is the one place a
        // value could plausibly leak into a file. It must not.
        let dir = tempfile::tempdir().unwrap();
        let paths = crate::paths::Paths::rooted(dir.path());
        let mut store = SecretStore::open(&paths);
        store.known.insert("github/personal-token".into());
        store.save_index().unwrap();

        let written = std::fs::read_to_string(paths.data_dir().join("secrets.json")).unwrap();
        assert!(written.contains("github/personal-token"));
        assert!(!written.contains("ghp_"));
    }

    #[test]
    fn the_index_survives_a_restart() {
        let dir = tempfile::tempdir().unwrap();
        let paths = crate::paths::Paths::rooted(dir.path());
        let mut store = SecretStore::open(&paths);
        store.known.insert("github/personal-token".into());
        store.save_index().unwrap();

        let reopened = SecretStore::open(&paths);
        assert_eq!(
            reopened.list(),
            vec![SecretRef::new("github", "personal-token")]
        );
    }

    /// The platform store itself is exercised only where there is one to
    /// exercise: a headless CI runner has no Keychain and no D-Bus session,
    /// and a test that needs one is a test that fails for the wrong reason.
    #[test]
    fn storing_and_retrieving_works_where_a_credential_store_exists() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = SecretStore::open(&crate::paths::Paths::rooted(dir.path()));
        let reference = SecretRef::new("cookie-test", uuid::Uuid::new_v4().to_string());

        match store.store(&reference, "hunter2") {
            Ok(()) => {
                assert_eq!(store.get(&reference).unwrap(), "hunter2");
                assert_eq!(
                    substitute(&store, &format!("token: {reference}")).unwrap(),
                    "token: hunter2"
                );
                assert!(store.forget(&reference).unwrap());
                assert!(!store.contains(&reference));
                // Forgetting something twice is not a failure.
                assert!(!store.forget(&reference).unwrap());
            }
            Err(e) => {
                eprintln!("no credential store on this machine, skipping: {e}");
            }
        }
    }
}
