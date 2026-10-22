//! Transport adapter for the official LigParGen Gradio service.
//! Called only by the remote-enabled OPLS provider; it supplies complete topologies.
//!
//! API: https://lsdodda-ligpargen.hf.space/gradio_api/info

use std::{
    io::{self, Cursor, Read},
    time::Duration,
};

use serde_json::{Value, json};

use super::ligpargen::Generated;

const BASE: &str = "https://lsdodda-ligpargen.hf.space";

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::other(message.into())
}

fn json_response(response: &mut ureq::http::Response<ureq::Body>) -> io::Result<Value> {
    let text = response
        .body_mut()
        .read_to_string()
        .map_err(io::Error::other)?;
    serde_json::from_str(&text).map_err(io::Error::other)
}

fn file_url(value: &Value) -> io::Result<String> {
    let path = value
        .get("url")
        .and_then(Value::as_str)
        .or_else(|| value.get("path").and_then(Value::as_str))
        .or_else(|| value.as_str())
        .ok_or_else(|| invalid("LigParGen did not return an output file"))?;
    // File downloads must stay on the provider's host.
    if path.starts_with(&format!("{BASE}/")) {
        Ok(path.into())
    } else if path.starts_with('/') && !path.starts_with("//") {
        Ok(format!("{BASE}/gradio_api/file={path}"))
    } else {
        Err(invalid("LigParGen returned an unexpected file URL"))
    }
}

pub(crate) fn parameterize(mol: &str, charge: i32) -> io::Result<Generated> {
    if !(-2..=2).contains(&charge) {
        return Err(invalid(
            "The public LigParGen service supports charges from -2 to +2",
        ));
    }
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(240)))
        .build()
        .into();
    let boundary = "dynamics-ligpargen-mol-upload";
    let body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"ligand.mol\"\r\nContent-Type: chemical/x-mdl-molfile\r\n\r\n{mol}\r\n--{boundary}--\r\n"
    );
    let mut uploaded = agent
        .post(format!("{BASE}/gradio_api/upload"))
        .header(
            "Content-Type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .send(body.as_bytes())
        .map_err(io::Error::other)?;
    let upload = json_response(&mut uploaded)?;
    let path = upload
        .get(0)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("LigParGen rejected the MOL upload"))?;
    let model = if charge == 0 {
        "1.14*CM1A-LBCC (neutral molecules)"
    } else {
        "1.14*CM1A (neutral or charged)"
    };
    let mut submitted = agent
        .post(format!("{BASE}/gradio_api/call/run_ligpargen"))
        .send_json(
            json!({"data": ["", {"path": path, "meta": {"_type": "gradio.FileData"}},
            "0", model, charge.to_string()]}),
        )
        .map_err(io::Error::other)?;
    let response = json_response(&mut submitted)?;
    let job_id = response
        .get("event_id")
        .and_then(Value::as_str)
        .filter(|id| id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-'))
        .ok_or_else(|| invalid("LigParGen did not return a job identifier"))?
        .to_string();
    let events = agent
        .get(format!("{BASE}/gradio_api/call/run_ligpargen/{job_id}"))
        .call()
        .map_err(io::Error::other)?
        .body_mut()
        .read_to_string()
        .map_err(io::Error::other)?;
    let mut event = "";
    let mut result = None;
    for line in events.lines() {
        if let Some(kind) = line.strip_prefix("event: ") {
            event = kind;
        }
        if let Some(data) = line.strip_prefix("data: ") {
            if event == "error" {
                let kind = if data.to_lowercase().contains("rate")
                    || data.to_lowercase().contains("limit")
                {
                    io::ErrorKind::WouldBlock
                } else {
                    io::ErrorKind::Other
                };
                return Err(io::Error::new(kind, format!("LigParGen: {data}")));
            }
            if event == "complete" {
                result = Some(serde_json::from_str::<Value>(data).map_err(io::Error::other)?);
            }
        }
    }
    let result = result.ok_or_else(|| invalid("LigParGen ended without a completed topology"))?;
    let archive_url = file_url(&result[0])?;
    let sdf_url = file_url(&result[1])?;
    if !sdf_url.to_lowercase().ends_with(".sdf") {
        return Err(invalid(
            "LigParGen did not return an SDF graph for output validation",
        ));
    }
    let sdf = agent
        .get(sdf_url)
        .call()
        .map_err(io::Error::other)?
        .body_mut()
        .read_to_string()
        .map_err(io::Error::other)?;
    let archive = agent
        .get(archive_url)
        .call()
        .map_err(io::Error::other)?
        .body_mut()
        .read_to_vec()
        .map_err(io::Error::other)?;
    let mut zip = zip::ZipArchive::new(Cursor::new(archive)).map_err(io::Error::other)?;
    let mut itp = None;
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i).map_err(io::Error::other)?;
        if entry.name().ends_with(".itp") {
            if itp.is_some() {
                return Err(invalid("LigParGen returned several molecule ITP files"));
            }
            let mut text = String::new();
            entry
                .by_ref()
                .take(8 * 1024 * 1024)
                .read_to_string(&mut text)?;
            itp = Some(text);
        }
    }
    Ok(Generated {
        sdf,
        itp: itp.ok_or_else(|| invalid("LigParGen archive has no GROMACS ITP"))?,
        job_id,
    })
}
