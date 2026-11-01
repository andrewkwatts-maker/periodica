//====== periodica/rust/periodica_core/src/pyjson.rs ======//
//!copyright (c) 2025 Andrew Keith Watts. All rights reserved.
//!
//!This is the intellectual property of Andrew Keith Watts. Unauthorized
//!reproduction, distribution, or modification of this code, in whole or in part,
//!without the express written permission of Andrew Keith Watts is strictly prohibited.
//!
//!For inquiries, please contact AndrewKWatts@Gmail.com

//! # pyjson
//!
//! JSON text exactly as Python's
//! `json.dump(obj, fp, indent=2, sort_keys=True, ensure_ascii=False)` writes
//! it -- the format of every generated datasheet under `data/derived/`.
//!
//! `Save` moved from Python to Rust; regenerating a tier must not rewrite
//! every file just because the writer changed. The differences from
//! `serde_json`'s pretty printer that matter are float spelling (Python's
//! `repr`: `1e-05`, `1e+16`, `2.0`, where Rust/serde print `1e-5`, `1e16`,
//! `2`... ) and key order (sorted). The writer is checked against every
//! committed derived file.

use serde_json::Value;

/// Serialise `value` the way Python's `json.dump(..., indent=2,
/// sort_keys=True, ensure_ascii=False)` does. No trailing newline.
pub fn to_string_sorted_indent2(value: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, value, 0);
    out
}

fn write_value(out: &mut String, value: &Value, level: usize) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.push_str(&i.to_string());
            } else if let Some(u) = n.as_u64() {
                out.push_str(&u.to_string());
            } else {
                out.push_str(&float_repr(n.as_f64().unwrap_or(f64::NAN)));
            }
        }
        Value::String(s) => write_string(out, s),
        Value::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                newline(out, level + 1);
                write_value(out, item, level + 1);
            }
            newline(out, level);
            out.push(']');
        }
        Value::Object(map) => {
            if map.is_empty() {
                out.push_str("{}");
                return;
            }
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, key) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                newline(out, level + 1);
                write_string(out, key);
                out.push_str(": ");
                write_value(out, &map[key], level + 1);
            }
            newline(out, level);
            out.push('}');
        }
    }
}

fn newline(out: &mut String, level: usize) {
    out.push('\n');
    for _ in 0..level {
        out.push_str("  ");
    }
}

/// Python's `ensure_ascii=False` string escaping: only `"`, `\` and the C0
/// control characters are escaped.
fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python's `repr(float)`: the shortest round-tripping digits, positional
/// for decimal exponents in [-4, 16), scientific (`1e-05`, `1.5e+16`)
/// otherwise, and always a `.0` on integral positional values.
pub fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "NaN".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity" } else { "-Infinity" }.to_string();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.to_string();
    }
    // Rust's `{:e}` prints the shortest round-tripping digits: "d.ddde<exp>".
    let sci = format!("{:e}", x.abs());
    let (mantissa, exp) = sci.split_once('e').expect("LowerExp always has an exponent");
    let exp: i32 = exp.parse().expect("integer exponent");
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let n = digits.len() as i32;
    // x = 0.<digits> * 10^decpt
    let decpt = exp + 1;

    let mut s = String::new();
    if x < 0.0 {
        s.push('-');
    }
    if decpt <= -4 || decpt > 16 {
        s.push_str(&digits[..1]);
        if n > 1 {
            s.push('.');
            s.push_str(&digits[1..]);
        }
        let e = decpt - 1;
        s.push_str(&format!("e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs()));
    } else if decpt <= 0 {
        s.push_str("0.");
        s.push_str(&"0".repeat((-decpt) as usize));
        s.push_str(&digits);
    } else if decpt >= n {
        s.push_str(&digits);
        s.push_str(&"0".repeat((decpt - n) as usize));
        s.push_str(".0");
    } else {
        s.push_str(&digits[..decpt as usize]);
        s.push('.');
        s.push_str(&digits[decpt as usize..]);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn floats_are_spelled_like_python_repr() {
        // Each right-hand side is CPython's repr() of the left.
        let cases: &[(f64, &str)] = &[
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (2.5, "2.5"),
            (-0.5, "-0.5"),
            (0.1, "0.1"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1.43e-07, "1.43e-07"),
            (2.9914611284799e-26, "2.9914611284799e-26"),
            (16780.8662550963, "16780.8662550963"),
            (55.406659999999995, "55.406659999999995"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1.5e16, "1.5e+16"),
            (123456789012345680.0, "1.2345678901234568e+17"),
            (1e100, "1e+100"),
            (1e-100, "1e-100"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (5e-324, "5e-324"),
            (0.3333333333, "0.3333333333"),
        ];
        for (x, want) in cases {
            assert_eq!(float_repr(*x), *want, "{x:e}");
        }
    }

    #[test]
    fn layout_matches_json_dump_indent2_sorted() {
        let v = json!({"b": [1, 2.0, {"z": null, "a": true}], "a": {}, "c": [], "é": "x\"\n\u{1}"});
        let want = "{\n  \"a\": {},\n  \"b\": [\n    1,\n    2.0,\n    {\n      \"a\": true,\n      \"z\": null\n    }\n  ],\n  \"c\": [],\n  \"é\": \"x\\\"\\n\\u0001\"\n}";
        assert_eq!(to_string_sorted_indent2(&v), want);
    }

    #[test]
    fn committed_derived_files_round_trip_byte_for_byte() {
        // Every generated datasheet was written by Python's json.dump; the
        // native writer must reproduce each one exactly, or a rebuild would
        // churn the whole tree.
        let derived =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../src/periodica/data/derived");
        let Ok(tiers) = std::fs::read_dir(&derived) else {
            return;
        };
        let mut checked = 0;
        for tier in tiers.flatten().filter(|e| e.path().is_dir()) {
            for file in std::fs::read_dir(tier.path()).unwrap().flatten() {
                let path = file.path();
                if path.extension().and_then(|x| x.to_str()) != Some("json") {
                    continue;
                }
                let text = std::fs::read_to_string(&path).unwrap().replace("\r\n", "\n");
                let value: Value = serde_json::from_str(&text).unwrap();
                assert_eq!(
                    to_string_sorted_indent2(&value),
                    text.trim_end_matches('\n'),
                    "{}",
                    path.display()
                );
                checked += 1;
            }
        }
        assert!(checked > 300, "{checked}");
    }
}
