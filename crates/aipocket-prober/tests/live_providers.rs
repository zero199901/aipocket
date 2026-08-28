//! Live smoke test across every registered provider spec.
//!
//! Ignored by default: it hits the real provider APIs with obviously-invalid
//! credentials, exactly like a scan's validation phase. Run with:
//!
//! ```bash
//! cargo test -p aipocket-prober --test live_providers -- --ignored --nocapture
//! ```

use aipocket_core::Credential;
use aipocket_prober::{ProviderRegistry, Validator};

fn dummy_key(prefixes: &[&str]) -> String {
    let prefix = prefixes.first().copied().unwrap_or("");
    format!("{prefix}live0000smoke0000key00000000000")
}

#[tokio::test]
#[ignore = "hits live provider APIs with invalid credentials"]
async fn live_validators_classify_all_provider_specs() {
    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(20))
        .build()
        .unwrap();
    let validator = Validator::new(http);
    let mut specs: Vec<_> = ProviderRegistry.specs().into_values().collect();
    specs.sort_by_key(|spec| spec.name);
    let mut rejected = 0;
    for spec in &specs {
        let result = validator
            .validate(Credential {
                apikey: dummy_key(spec.key_prefixes),
                apiurl: spec.official_api_url.into(),
                ..Default::default()
            })
            .await
            .unwrap_or_else(|error| panic!("{} validate failed: {error}", spec.name));
        let status = result
            .status_code
            .map_or_else(|| "-".to_string(), |code| code.to_string());
        println!(
            "{:<14} http={:<4} state={:<14} error={}",
            spec.name, status, result.validation_state, result.error
        );
        assert!(
            matches!(
                result.validation_state.as_str(),
                "final_verified" | "rejected" | "transient"
            ),
            "{}: unexpected state {}",
            spec.name,
            result.validation_state
        );
        if matches!(result.status_code, Some(401 | 403)) {
            assert_eq!(
                result.validation_state, "rejected",
                "{}: definitive HTTP rejection misfiled",
                spec.name
            );
        }
        rejected += u8::from(result.validation_state == "rejected");
    }
    println!("\n{}/{} specs classified rejected", rejected, specs.len());
}

#[tokio::test]
#[ignore = "hits the live Gemini API"]
async fn live_gemini_invalid_key_is_rejected_not_transient() {
    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(20))
        .build()
        .unwrap();
    let validator = Validator::new(http);
    let result = validator
        .validate(Credential {
            apikey: "AIzaSyA0000000000000000000000000000000000".into(),
            apiurl: "https://generativelanguage.googleapis.com".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    println!(
        "gemini http={:?} state={} error={} snippet={}",
        result.status_code,
        result.validation_state,
        result.error,
        result
            .response_snippet
            .chars()
            .take(160)
            .collect::<String>()
    );
    // Google answers invalid keys with 400 API_KEY_INVALID; the fix under
    // test must classify that as a definitive rejection.
    if result.status_code == Some(400) {
        assert_eq!(result.validation_state, "rejected");
        assert_eq!(result.error, "unauthorized");
    }
}
