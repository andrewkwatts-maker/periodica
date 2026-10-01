//====== periodica/rust/periodica-mat/src/jsonc.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # jsonc
//!
//! periodica's datasheets are **JSONC**, not JSON. 36 of them carry `//`
//! comments:
//!
//! ```text
//! "Charge_e": 0.6666666667, // +2/3 e
//! "Mass_MeVc2": 2.2,        // Current quark mass, approximate
//! ```
//!
//! Every Python reader strips them before parsing -- `get.py:98`, `get.py:158`,
//! `data/data_manager.py:137`, `data/quark_loader.py:144` and
//! `data/quark_source.py:121` all run the same
//! `re.sub(r"//.*?(?=\n|$)", "", text)`. The Rust `data_loader` did not, which
//! is a second reason (beyond the tier-name mismatch) that it silently loaded
//! almost nothing: `serde_json::from_slice` rejects all 36 files with
//! "key must be a string".
//!
//! ## Why not just copy the Python regex
//!
//! `//.*?(?=\n|$)` is string-blind: it would happily truncate
//! `"url": "https://example.com"` to `"url": "https`, producing invalid JSON.
//! No bundled datasheet trips that today, but the app is about to let designers
//! author material files, and a URL or a Windows path in a description field is
//! entirely plausible.
//!
//! [`strip_comments`] is a proper single-pass scanner that tracks string state
//! and escapes. It accepts a strict superset of what the Python regex accepts,
//! so parity is preserved on all existing data while the failure mode is
//! removed.

/// Remove `//` line comments and `/* ... */` block comments from JSONC text,
/// leaving anything inside a string literal untouched.
///
/// Comment bodies are replaced with nothing, but newlines inside block comments
/// are preserved so that `serde_json`'s line numbers in an error still point at
/// the right line.
pub fn strip_comments(src: &str) -> String {
    // Scanning at the byte level is both simpler and safe for UTF-8: every
    // character this scanner reacts to (`"`, `/`, `*`, `\`, `\n`) is ASCII, and
    // no continuation byte of a multi-byte sequence is ever ASCII. Comment
    // regions therefore always begin and end on a character boundary, so
    // copying bytes verbatim can never split or reinterpret a code point.
    //
    // (Converting bytes to `char` individually -- the obvious-looking version
    // of this loop -- silently turns "café" into "cafÃ©".)
    let bytes = src.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(src.len());
    let mut i = 0usize;
    let mut in_string = false;

    while i < bytes.len() {
        let b = bytes[i];

        if in_string {
            out.push(b);
            if b == b'\\' {
                // Copy the escaped byte verbatim so `\"` cannot end the string.
                if i + 1 < bytes.len() {
                    out.push(bytes[i + 1]);
                    i += 2;
                    continue;
                }
            } else if b == b'"' {
                in_string = false;
            }
            i += 1;
            continue;
        }

        match b {
            b'"' => {
                in_string = true;
                out.push(b'"');
                i += 1;
            }
            b'/' if i + 1 < bytes.len() && bytes[i + 1] == b'/' => {
                // Line comment: skip to end of line, keeping the newline.
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            b'/' if i + 1 < bytes.len() && bytes[i + 1] == b'*' => {
                // Block comment: skip to the terminator, preserving newlines so
                // that parse-error line numbers still point at the right line.
                i += 2;
                while i < bytes.len() {
                    if bytes[i] == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                        i += 2;
                        break;
                    }
                    if bytes[i] == b'\n' {
                        out.push(b'\n');
                    }
                    i += 1;
                }
            }
            _ => {
                out.push(b);
                i += 1;
            }
        }
    }

    // Guaranteed to succeed by the boundary argument above; `_lossy` is a
    // belt-and-braces fallback rather than an expected path.
    String::from_utf8(out).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned())
}

/// Parse JSONC text into a [`serde_json::Value`].
///
/// Plain JSON is parsed directly; the comment-stripping pass only runs if that
/// first attempt fails, so the common case costs nothing.
pub fn from_str(text: &str) -> Result<serde_json::Value, serde_json::Error> {
    match serde_json::from_str(text) {
        Ok(v) => Ok(v),
        Err(_) => serde_json::from_str(&strip_comments(text)),
    }
}

/// Read a `.json`/`.jsonc` file from disk and parse it.
pub fn from_path(path: &std::path::Path) -> std::io::Result<serde_json::Value> {
    let text = std::fs::read_to_string(path)?;
    from_str(&text).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn strips_trailing_line_comments() {
        // The exact shape found in active/quarks/UpQuark.json.
        let src = r#"{
  "Name": "Up Quark",
  "Charge_e": 0.6666666667, // +2/3 e
  "Mass_MeVc2": 2.2, // Current quark mass, approximate
  "MagneticDipoleMoment_J_T": null // Complex, not a fixed value
}"#;
        let v = from_str(src).unwrap();
        assert_eq!(v["Name"], "Up Quark");
        assert_eq!(v["Mass_MeVc2"], 2.2);
        assert!(v["MagneticDipoleMoment_J_T"].is_null());
    }

    #[test]
    fn strips_whole_line_and_block_comments() {
        let src = r#"{
  // a leading comment
  "a": 1,
  /* a block
     spanning lines */
  "b": 2
}"#;
        assert_eq!(from_str(src).unwrap(), json!({"a": 1, "b": 2}));
    }

    #[test]
    fn does_not_touch_slashes_inside_strings() {
        // This is where the Python regex is wrong: it would truncate the URL
        // and produce invalid JSON.
        let src = r#"{"url": "https://example.com/a//b", "path": "C:\\x//y"}"#;
        let v = from_str(src).unwrap();
        assert_eq!(v["url"], "https://example.com/a//b");
        assert_eq!(v["path"], r"C:\x//y");
    }

    #[test]
    fn python_regex_would_have_broken_this_case() {
        // Demonstrate the improvement concretely rather than asserting intent.
        let src = r#"{"u": "http://x.com"}"#;
        let naive = regex_like_python(src);
        assert!(
            serde_json::from_str::<serde_json::Value>(&naive).is_err(),
            "the naive strip should corrupt this"
        );
        assert_eq!(from_str(src).unwrap()["u"], "http://x.com");
    }

    /// Mimic `re.sub(r"//.*?(?=\n|$)", "", text)` for the comparison above.
    fn regex_like_python(s: &str) -> String {
        s.lines()
            .map(|l| match l.find("//") {
                Some(i) => &l[..i],
                None => l,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn escaped_quotes_do_not_end_a_string() {
        let src = r#"{"s": "he said \"hi\" // not a comment"}"#;
        assert_eq!(
            from_str(src).unwrap()["s"],
            r#"he said "hi" // not a comment"#
        );
    }

    #[test]
    fn plain_json_is_unchanged() {
        let src = r#"{"a":[1,2,3],"b":{"c":true},"d":1.5e-7}"#;
        assert_eq!(
            from_str(src).unwrap(),
            serde_json::from_str::<serde_json::Value>(src).unwrap()
        );
        assert_eq!(strip_comments(src), src);
    }

    #[test]
    fn utf8_survives_stripping() {
        let src = "{\"n\": \"caf\u{e9} \u{2014} \u{1f600}\", \"v\": 1} // tail";
        let v = from_str(src).unwrap();
        assert_eq!(v["n"], "caf\u{e9} \u{2014} \u{1f600}");
        assert_eq!(v["v"], 1);
    }

    #[test]
    fn line_numbers_are_preserved_across_block_comments() {
        // A parse error after a multi-line block comment should still report a
        // useful line, so the newlines must survive.
        let src = "{\n/* one\ntwo\nthree */\n  \"a\": ,\n}";
        let err = from_str(src).unwrap_err();
        assert_eq!(err.line(), 5, "got {err}");
    }

    #[test]
    fn unterminated_string_does_not_loop_or_panic() {
        assert!(from_str("{\"a\": \"unterminated").is_err());
        assert!(from_str("{\"a\": 1 /* unterminated").is_err());
        assert!(from_str("\\").is_err());
    }
}
