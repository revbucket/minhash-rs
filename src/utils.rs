use anyhow::{anyhow, Error, Result};
use serde_json::{json, Value};

pub fn json_get<'a>(data: &'a serde_json::Value, key: &str) -> Option<&'a Value> {
    let keys: Vec<&str> = key.split('.').collect();
    let mut current = data;

    for key in keys {
        match current.get(key) {
            Some(value) => current = value,
            None => return None,
        }
    }

    Some(current)
}

pub fn json_set(input: &mut Value, key: &String, val: Value) -> Result<(), Error> {
    let parts: Vec<&str> = key.split('.').collect();
    let mut current = input;

    for (i, &part) in parts.iter().enumerate() {
        if i == parts.len() - 1 {
            if current.is_object() {
                current[part] = val;
                return Ok(());
            } else {
                return Err(anyhow!("Weird nesting for setting json values"));
            }
        }
        if !current.is_object() {
            return Err(anyhow!("Weird nesting for setting json values"));
        }
        if !current.get(part).is_some() {
            current[part] = json!({});
        }
        current = &mut current[part];
    }
    Ok(())
}
