use aipocket_core::Credential;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CredentialKind {
    Standard,
    Admin,
    ServiceAccount,
    OAuth,
    Unknown,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SpecializedValidation {
    pub valid: bool,
    pub status_code: Option<u16>,
    pub credential_kind: String,
    pub scope: String,
    pub tier_evidence: String,
    pub models: Vec<String>,
    pub error: String,
    pub evidence: Value,
}

pub fn classify_credential(provider: &str, key: &str) -> CredentialKind {
    match provider {
        "openai" if key.starts_with("sk-admin-") => CredentialKind::Admin,
        "openai" if key.starts_with("sk-svcacct-") => CredentialKind::ServiceAccount,
        "anthropic" if key.starts_with("sk-ant-admin") => CredentialKind::Admin,
        "anthropic" if key.starts_with("sk-ant-oat") || key.starts_with("sk-ant-sid") => {
            CredentialKind::OAuth
        }
        _ if key.is_empty() => CredentialKind::Unknown,
        _ => CredentialKind::Standard,
    }
}

pub async fn validate_specialized(
    http: &reqwest::Client,
    credential: &Credential,
    provider: &str,
) -> anyhow::Result<Option<SpecializedValidation>> {
    let kind = classify_credential(provider, &credential.apikey);
    let routed_apiurl = routed_apiurl(credential, provider);
    let base = routed_apiurl.trim_end_matches('/');
    let origin = base
        .split("/v1beta")
        .next()
        .unwrap_or(base)
        .trim_end_matches('/');
    let response = match provider {
        "anthropic" => Some(
            http.get(if base.ends_with("/v1") {
                format!("{base}/models")
            } else {
                format!("{base}/v1/models")
            })
            .header("x-api-key", &credential.apikey)
            .header("anthropic-version", "2023-06-01")
            .send()
            .await?,
        ),
        "gemini" => Some(
            http.get(format!("{origin}/v1beta/models"))
                .query(&[("key", &credential.apikey)])
                .send()
                .await?,
        ),
        "azure_openai" => Some(
            http.get(format!("{base}/openai/models?api-version=2024-10-21"))
                .header("api-key", &credential.apikey)
                .send()
                .await?,
        ),
        "openai" => Some(
            http.get(if base.ends_with("/v1") {
                format!("{base}/models")
            } else {
                format!("{base}/v1/models")
            })
            .bearer_auth(&credential.apikey)
            .send()
            .await?,
        ),
        "qoder" => Some(
            http.get(if base.ends_with("/api/v1") {
                format!("{base}/cloud/models")
            } else {
                format!("{base}/api/v1/cloud/models")
            })
            .bearer_auth(&credential.apikey)
            .send()
            .await?,
        ),
        "cursor" => Some(
            http.get(if base.ends_with("/v1") {
                format!("{base}/me")
            } else {
                format!("{base}/v1/me")
            })
            .bearer_auth(&credential.apikey)
            .send()
            .await?,
        ),
        "openrouter" => {
            let origin = base
                .split("/api")
                .next()
                .unwrap_or(base)
                .trim_end_matches('/');
            let response = http
                .get(format!("{origin}/api/v1/auth/key"))
                .bearer_auth(&credential.apikey)
                .send()
                .await?;
            if !response.status().is_success() {
                Some(response)
            } else {
                Some(
                    http.get(format!("{origin}/api/v1/models"))
                        .bearer_auth(&credential.apikey)
                        .send()
                        .await?,
                )
            }
        }
        "aws_bedrock" => Some(
            http.get(if base.ends_with("/foundation-models") {
                base.to_owned()
            } else {
                format!("{base}/foundation-models")
            })
            .bearer_auth(&credential.apikey)
            .send()
            .await?,
        ),
        _ => None,
    };
    let Some(response) = response else {
        return Ok(None);
    };
    let status = response.status();
    let body: Value = response.json().await.unwrap_or(json!({}));
    let models = extract_models(&body);
    let provider_evidence = valid_provider_evidence(provider, &body);
    let valid = status.is_success() && provider_evidence;
    Ok(Some(SpecializedValidation {
        valid,
        status_code: Some(status.as_u16()),
        credential_kind: format!("{kind:?}").to_ascii_lowercase(),
        scope: body
            .get("organization_id")
            .or_else(|| body.get("account_id"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .into(),
        tier_evidence: body
            .get("tier")
            .or_else(|| body.get("plan"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .into(),
        models,
        error: if status.as_u16() == 401
            || status.as_u16() == 403
            || crate::validator::is_definitive_key_rejection(status.as_u16(), &body)
        {
            "unauthorized".into()
        } else if !status.is_success() {
            "read-failed".into()
        } else if !provider_evidence {
            "invalid-response-schema".into()
        } else {
            String::new()
        },
        evidence: body,
    }))
}
fn routed_apiurl(credential: &Credential, provider: &str) -> String {
    match provider {
        "openai" | "anthropic" | "gemini" | "openrouter"
            if provider_from_key(provider, &credential.apikey) =>
        {
            match provider {
                "openai" => "https://api.openai.com/v1",
                "anthropic" => "https://api.anthropic.com/v1",
                "gemini" => "https://generativelanguage.googleapis.com",
                "openrouter" => "https://openrouter.ai/api",
                _ => unreachable!(),
            }
            .into()
        }
        _ => credential.apiurl.clone(),
    }
}

fn provider_from_key(provider: &str, key: &str) -> bool {
    match provider {
        "openai" => {
            key.starts_with("sk-proj-")
                || key.starts_with("sk-admin-")
                || key.starts_with("sk-svcacct-")
        }
        "anthropic" => key.starts_with("sk-ant-"),
        "gemini" => key.starts_with("AIza"),
        "openrouter" => key.starts_with("sk-or-"),
        _ => false,
    }
}

fn valid_provider_evidence(provider: &str, body: &Value) -> bool {
    match provider {
        "cursor" => {
            body.get("apiKeyName").and_then(Value::as_str).is_some()
                || body.get("userEmail").and_then(Value::as_str).is_some()
        }
        _ => !extract_models(body).is_empty(),
    }
}

fn extract_models(value: &Value) -> Vec<String> {
    value
        .get("data")
        .or_else(|| value.get("models"))
        .or_else(|| value.get("modelSummaries"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|item| {
            item.get("id")
                .or_else(|| item.get("name"))
                .or_else(|| item.get("modelId"))
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .collect()
}
#[cfg(test)]
mod validation_tests {
    use super::*;
    use axum::{Json, Router, http::StatusCode, routing::get};

    #[tokio::test]
    async fn openrouter_requires_authenticated_key_before_models() {
        async fn denied() -> (StatusCode, Json<Value>) {
            (StatusCode::UNAUTHORIZED, Json(json!({"error":"denied"})))
        }
        let app = Router::new().route("/api/v1/auth/key", get(denied));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let result = validate_specialized(
            &reqwest::Client::new(),
            &Credential {
                apikey: "not-prefix-routed".into(),
                apiurl: format!("http://{address}/api"),
                ..Default::default()
            },
            "openrouter",
        )
        .await
        .unwrap()
        .unwrap();
        assert!(!result.valid);
        assert_eq!(result.status_code, Some(401));
        assert_eq!(result.error, "unauthorized");
        server.abort();
    }

    #[tokio::test]
    async fn rejects_success_without_provider_evidence() {
        let app = Router::new().route(
            "/v1/models",
            get(|| async { (StatusCode::OK, "<!doctype html><title>fallback</title>") }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let result = validate_specialized(
            &reqwest::Client::new(),
            &Credential {
                apikey: "not-prefix-routed".into(),
                apiurl: format!("http://{address}"),
                ..Default::default()
            },
            "openai",
        )
        .await
        .unwrap()
        .unwrap();
        assert!(!result.valid);
        assert_eq!(result.status_code, Some(200));
        assert_eq!(result.error, "invalid-response-schema");
        server.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn classifies_admin_and_oauth() {
        assert_eq!(
            classify_credential("openai", "sk-admin-x"),
            CredentialKind::Admin
        );
        assert_eq!(
            classify_credential("anthropic", "sk-ant-oat-x"),
            CredentialKind::OAuth
        );
    }
}
