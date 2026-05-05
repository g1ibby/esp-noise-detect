use std::collections::BTreeMap;
use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use burn_espdl_export::{EspdlFile, dl};

const USAGE: &str = "\
usage:
  cargo xtask espdl-acceptance \\
    --checkpoint <path.mpk|stem> \\
    --config <yaml> \\
    --manifest <manifest.jsonl> \\
    [--out-dir target/espdl-acceptance] \\
    [--backend metal] \\
    [--target esp32s3] \\
    [--num-bits 8] \\
    [--calib-split train|val] \\
    [--calib-windows 512] \\
    [--calib-steps 64]
";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args_os();
    let _bin = args.next();
    let Some(cmd) = args.next() else {
        return Err(USAGE.to_string());
    };
    match cmd.to_string_lossy().as_ref() {
        "espdl-acceptance" => espdl_acceptance(parse_acceptance_args(args)?),
        "-h" | "--help" => {
            println!("{USAGE}");
            Ok(())
        }
        other => Err(format!("unknown xtask subcommand {other:?}\n{USAGE}")),
    }
}

#[derive(Debug)]
struct AcceptanceArgs {
    checkpoint: PathBuf,
    config: PathBuf,
    manifest: PathBuf,
    out_dir: PathBuf,
    backend: String,
    target: String,
    num_bits: u8,
    calib_split: String,
    calib_windows: usize,
    calib_steps: usize,
}

fn parse_acceptance_args<I>(args: I) -> Result<AcceptanceArgs, String>
where
    I: IntoIterator<Item = OsString>,
{
    let mut checkpoint = None;
    let mut config = None;
    let mut manifest = None;
    let mut out_dir = PathBuf::from("target/espdl-acceptance");
    let mut backend = "metal".to_string();
    let mut target = "esp32s3".to_string();
    let mut num_bits = 8_u8;
    let mut calib_split = "train".to_string();
    let mut calib_windows = 512_usize;
    let mut calib_steps = 64_usize;

    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let arg = arg.to_string_lossy();
        match arg.as_ref() {
            "--checkpoint" => checkpoint = Some(PathBuf::from(need_os(&mut args, "--checkpoint")?)),
            "--config" => config = Some(PathBuf::from(need_os(&mut args, "--config")?)),
            "--manifest" => manifest = Some(PathBuf::from(need_os(&mut args, "--manifest")?)),
            "--out-dir" => out_dir = PathBuf::from(need_os(&mut args, "--out-dir")?),
            "--backend" => backend = need_string(&mut args, "--backend")?,
            "--target" => target = need_string(&mut args, "--target")?,
            "--num-bits" => num_bits = parse_value(&mut args, "--num-bits")?,
            "--calib-split" => calib_split = need_string(&mut args, "--calib-split")?,
            "--calib-windows" => calib_windows = parse_value(&mut args, "--calib-windows")?,
            "--calib-steps" => calib_steps = parse_value(&mut args, "--calib-steps")?,
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => {
                return Err(format!(
                    "unknown espdl-acceptance argument {other:?}\n{USAGE}"
                ));
            }
        }
    }

    if !matches!(backend.as_str(), "metal" | "cuda" | "vulkan" | "webgpu") {
        return Err(format!(
            "unsupported --backend {backend:?}; use metal, cuda, vulkan, or webgpu"
        ));
    }
    if num_bits != 8 {
        return Err("--num-bits must be 8; INT16 native export is not parity-verified".to_string());
    }
    if !matches!(
        calib_split.as_str(),
        "train" | "val" | "valid" | "validation"
    ) {
        return Err("--calib-split must be train or val".to_string());
    }
    if calib_windows == 0 {
        return Err("--calib-windows must be greater than zero".to_string());
    }
    if calib_steps == 0 {
        return Err("--calib-steps must be greater than zero".to_string());
    }

    Ok(AcceptanceArgs {
        checkpoint: checkpoint.ok_or_else(|| format!("missing --checkpoint\n{USAGE}"))?,
        config: config.ok_or_else(|| format!("missing --config\n{USAGE}"))?,
        manifest: manifest.ok_or_else(|| format!("missing --manifest\n{USAGE}"))?,
        out_dir,
        backend,
        target,
        num_bits,
        calib_split: normalize_split(&calib_split).to_string(),
        calib_windows,
        calib_steps,
    })
}

fn need_os<I>(args: &mut I, name: &str) -> Result<OsString, String>
where
    I: Iterator<Item = OsString>,
{
    args.next()
        .ok_or_else(|| format!("{name} takes an argument"))
}

fn need_string<I>(args: &mut I, name: &str) -> Result<String, String>
where
    I: Iterator<Item = OsString>,
{
    Ok(need_os(args, name)?.to_string_lossy().into_owned())
}

fn parse_value<I, T>(args: &mut I, name: &str) -> Result<T, String>
where
    I: Iterator<Item = OsString>,
    T: std::str::FromStr,
{
    need_string(args, name)?
        .parse()
        .map_err(|_| format!("{name} has an invalid value"))
}

fn normalize_split(split: &str) -> &str {
    match split {
        "valid" | "validation" => "val",
        other => other,
    }
}

fn espdl_acceptance(args: AcceptanceArgs) -> Result<(), String> {
    require_file(&args.checkpoint, "--checkpoint")?;
    require_file(&args.config, "--config")?;
    require_file(&args.manifest, "--manifest")?;
    require_file(
        Path::new("scripts/burn_to_espdl_legacy.sh"),
        "legacy exporter",
    )?;
    require_command("uv", "legacy Python export uses `uv run --project nn`")?;

    let native_dir = args.out_dir.join("native");
    let legacy_dir = args.out_dir.join("legacy");
    fs::create_dir_all(&native_dir).map_err(|e| format!("create {}: {e}", native_dir.display()))?;
    fs::create_dir_all(&legacy_dir).map_err(|e| format!("create {}: {e}", legacy_dir.display()))?;

    let mut report = Report::default();
    report.line("ESP-DL acceptance");
    report.line("===================");
    report.line(format!("checkpoint : {}", args.checkpoint.display()));
    report.line(format!("config     : {}", args.config.display()));
    report.line(format!("manifest   : {}", args.manifest.display()));
    report.line(format!("out-dir    : {}", args.out_dir.display()));
    report.line(format!(
        "calibration: native/legacy compare on first {} window(s); legacy dumped {}",
        args.calib_steps.min(args.calib_windows),
        args.calib_windows
    ));
    report.line("");

    let legacy_cmd = legacy_command(&args, &legacy_dir);
    report.line(format!("legacy command: {}", render_command(&legacy_cmd)));
    run_command("legacy exporter", legacy_cmd)?;
    let native_cmd = native_command(&args, &native_dir, Some(&legacy_dir.join("calib")));
    report.line(format!("native command: {}", render_command(&native_cmd)));
    run_command("native exporter", native_cmd)?;

    let native_model = native_dir.join("model.espdl");
    let legacy_model = legacy_dir.join("model.espdl");
    report.line("");
    report.line(format!("native model: {}", native_model.display()));
    report.line(format!("legacy model: {}", legacy_model.display()));

    let native_bytes =
        fs::read(&native_model).map_err(|e| format!("read {}: {e}", native_model.display()))?;
    let legacy_bytes =
        fs::read(&legacy_model).map_err(|e| format!("read {}: {e}", legacy_model.display()))?;
    let native_file = EspdlFile::parse(&native_bytes)
        .map_err(|e| format!("parse native {}: {e}", native_model.display()))?;
    let legacy_file = EspdlFile::parse(&legacy_bytes)
        .map_err(|e| format!("parse legacy {}: {e}", legacy_model.display()))?;

    let summary = compare_files(&native_file, &legacy_file);
    summary.append_to(&mut report);

    let report_path = args.out_dir.join("acceptance_report.txt");
    fs::write(&report_path, report.as_text())
        .map_err(|e| format!("write {}: {e}", report_path.display()))?;
    print!("{}", report.as_text());
    if summary.failures.is_empty() {
        println!("report: {}", report_path.display());
        Ok(())
    } else {
        Err(format!(
            "ESPDL acceptance failed with {} issue(s); report: {}",
            summary.failures.len(),
            report_path.display()
        ))
    }
}

fn require_file(path: &Path, label: &str) -> Result<(), String> {
    if path.is_file() {
        Ok(())
    } else {
        Err(format!("{label} not found: {}", path.display()))
    }
}

fn require_command(bin: &str, purpose: &str) -> Result<(), String> {
    let output = Command::new(bin)
        .arg("--version")
        .output()
        .map_err(|e| format!("missing prerequisite `{bin}` for {purpose}: {e}"))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "missing prerequisite `{bin}` for {purpose}: `{bin} --version` exited with {}",
            output.status
        ))
    }
}

fn native_command(args: &AcceptanceArgs, native_dir: &Path, calib_dir: Option<&Path>) -> Command {
    let mut cmd = Command::new(cargo_bin());
    cmd.arg("run")
        .arg("--release")
        .arg("-p")
        .arg("nn-rs")
        .arg("--no-default-features")
        .arg("--features")
        .arg(format!("std,{}", args.backend))
        .arg("--bin")
        .arg("export_espdl")
        .arg("--")
        .arg("--checkpoint")
        .arg(&args.checkpoint)
        .arg("--config")
        .arg(&args.config)
        .arg("--manifest")
        .arg(&args.manifest)
        .arg("--out-dir")
        .arg(native_dir)
        .arg("--target")
        .arg(&args.target)
        .arg("--num-bits")
        .arg(args.num_bits.to_string())
        .arg("--calib-split")
        .arg(&args.calib_split)
        .arg("--calib-windows")
        .arg(args.calib_steps.min(args.calib_windows).to_string());
    if let Some(calib_dir) = calib_dir {
        cmd.arg("--calib-dir").arg(calib_dir);
    }
    cmd
}

fn legacy_command(args: &AcceptanceArgs, legacy_dir: &Path) -> Command {
    let mut cmd = Command::new("scripts/burn_to_espdl_legacy.sh");
    cmd.arg("--checkpoint")
        .arg(&args.checkpoint)
        .arg("--config")
        .arg(&args.config)
        .arg("--manifest")
        .arg(&args.manifest)
        .arg("--out-dir")
        .arg(legacy_dir)
        .arg("--target")
        .arg(&args.target)
        .arg("--num-bits")
        .arg(args.num_bits.to_string())
        .arg("--calib-split")
        .arg(&args.calib_split)
        .arg("--calib-windows")
        .arg(args.calib_windows.to_string())
        .arg("--calib-steps")
        .arg(args.calib_steps.to_string());
    cmd
}

fn cargo_bin() -> OsString {
    env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"))
}

fn render_command(cmd: &Command) -> String {
    let mut parts = vec![shellish(cmd.get_program())];
    parts.extend(cmd.get_args().map(shellish));
    parts.join(" ")
}

fn shellish(s: &std::ffi::OsStr) -> String {
    let s = s.to_string_lossy();
    if s.chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '/' | ':' | ','))
    {
        s.into_owned()
    } else {
        format!("{s:?}")
    }
}

fn run_command(label: &str, mut cmd: Command) -> Result<(), String> {
    let status = cmd
        .status()
        .map_err(|e| format!("failed to start {label}: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{label} exited with {status}"))
    }
}

#[derive(Default)]
struct Report {
    lines: Vec<String>,
}

impl Report {
    fn line(&mut self, line: impl Into<String>) {
        self.lines.push(line.into());
    }

    fn as_text(&self) -> String {
        let mut text = self.lines.join("\n");
        text.push('\n');
        text
    }
}

#[derive(Default)]
struct CompareSummary {
    failures: Vec<String>,
    tensor_reports: Vec<String>,
    node_count: usize,
    initializer_count: usize,
}

impl CompareSummary {
    fn fail(&mut self, msg: impl Into<String>) {
        self.failures.push(msg.into());
    }

    fn append_to(&self, report: &mut Report) {
        report.line("");
        report.line("Comparison");
        report.line("----------");
        report.line(format!("nodes        : {}", self.node_count));
        report.line(format!("initializers : {}", self.initializer_count));
        for line in &self.tensor_reports {
            report.line(line);
        }
        report.line("");
        if self.failures.is_empty() {
            report.line("ESPDL acceptance: PASS");
        } else {
            report.line("ESPDL acceptance: FAIL");
            for failure in &self.failures {
                report.line(format!("- {failure}"));
            }
        }
    }
}

fn compare_files(native: &EspdlFile<'_>, legacy: &EspdlFile<'_>) -> CompareSummary {
    let mut summary = CompareSummary::default();
    let Some(native_graph) = native.model().graph() else {
        summary.fail("native model has no graph");
        return summary;
    };
    let Some(legacy_graph) = legacy.model().graph() else {
        summary.fail("legacy model has no graph");
        return summary;
    };

    let native_nodes = node_vec(native_graph.node());
    let legacy_nodes = node_vec(legacy_graph.node());
    compare_nodes(&mut summary, &native_nodes, &legacy_nodes);
    compare_value_info_sequence(
        &mut summary,
        "graph input",
        value_info_vec(native_graph.input()),
        value_info_vec(legacy_graph.input()),
    );
    compare_value_info_sequence(
        &mut summary,
        "graph output",
        value_info_vec(native_graph.output()),
        value_info_vec(legacy_graph.output()),
    );
    compare_node_output_value_infos(
        &mut summary,
        &native_nodes,
        &legacy_nodes,
        &value_info_map(native_graph.value_info()),
        &value_info_map(legacy_graph.value_info()),
        &value_info_map(native_graph.output()),
        &value_info_map(legacy_graph.output()),
    );
    compare_initializers(
        &mut summary,
        &native_nodes,
        &legacy_nodes,
        tensor_map(native_graph.initializer()),
        tensor_map(legacy_graph.initializer()),
    );
    summary
}

fn compare_nodes(summary: &mut CompareSummary, native: &[NodeSnap], legacy: &[NodeSnap]) {
    summary.node_count = native.len();
    if native.len() != legacy.len() {
        summary.fail(format!(
            "node count mismatch: native {} legacy {}",
            native.len(),
            legacy.len()
        ));
        return;
    }
    for (idx, (n, l)) in native.iter().zip(legacy.iter()).enumerate() {
        cmp_field(
            summary,
            format!("node {idx} op_type"),
            n.op_type.as_ref(),
            l.op_type.as_ref(),
        );
        cmp_field(
            summary,
            format!("node {idx} input count"),
            n.inputs.len(),
            l.inputs.len(),
        );
        cmp_field(
            summary,
            format!("node {idx} output count"),
            n.outputs.len(),
            l.outputs.len(),
        );
        cmp_field(
            summary,
            format!("node {idx} attributes"),
            &n.attributes,
            &l.attributes,
        );
    }
}

#[derive(Debug, PartialEq, Eq)]
struct NodeSnap {
    name: Option<String>,
    op_type: Option<String>,
    domain: Option<String>,
    inputs: Vec<String>,
    outputs: Vec<String>,
    attributes: BTreeMap<String, String>,
}

fn node_vec(
    nodes: Option<flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<dl::Node<'_>>>>,
) -> Vec<NodeSnap> {
    nodes
        .map(|nodes| {
            (0..nodes.len())
                .map(|i| {
                    let n = nodes.get(i);
                    NodeSnap {
                        name: n.name().map(str::to_string),
                        op_type: n.op_type().map(str::to_string),
                        domain: n.domain().map(str::to_string),
                        inputs: str_vec(n.input()),
                        outputs: str_vec(n.output()),
                        attributes: attr_map(n.attribute()),
                    }
                })
                .collect()
        })
        .unwrap_or_default()
}

fn attr_map(
    attrs: Option<flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<dl::Attribute<'_>>>>,
) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    if let Some(attrs) = attrs {
        for i in 0..attrs.len() {
            let attr = attrs.get(i);
            let key = attr
                .name()
                .map(str::to_string)
                .unwrap_or_else(|| format!("<unnamed:{i}>"));
            if key == "kernel_shape" {
                continue;
            }
            out.insert(key, format!("{attr:?}"));
        }
    }
    out
}

fn compare_value_info_sequence(
    summary: &mut CompareSummary,
    label: &str,
    native: Vec<ValueInfoSnap>,
    legacy: Vec<ValueInfoSnap>,
) {
    if native.len() != legacy.len() {
        summary.fail(format!(
            "{label} count mismatch: native {} legacy {}",
            native.len(),
            legacy.len()
        ));
        return;
    }
    for (idx, (n, l)) in native.iter().zip(&legacy).enumerate() {
        cmp_field(
            summary,
            format!("{label} {idx} type"),
            &n.type_info,
            &l.type_info,
        );
        cmp_field(
            summary,
            format!("{label} {idx} exponents"),
            &n.exponents,
            &l.exponents,
        );
    }
}

fn compare_node_output_value_infos(
    summary: &mut CompareSummary,
    native_nodes: &[NodeSnap],
    legacy_nodes: &[NodeSnap],
    native_values: &BTreeMap<String, ValueInfoSnap>,
    legacy_values: &BTreeMap<String, ValueInfoSnap>,
    native_outputs: &BTreeMap<String, ValueInfoSnap>,
    legacy_outputs: &BTreeMap<String, ValueInfoSnap>,
) {
    for (idx, (native_node, legacy_node)) in native_nodes.iter().zip(legacy_nodes).enumerate() {
        let Some(native_output) = native_node.outputs.first() else {
            continue;
        };
        let Some(legacy_output) = legacy_node.outputs.first() else {
            continue;
        };
        let native_value = native_values
            .get(native_output)
            .or_else(|| native_outputs.get(native_output));
        let legacy_value = legacy_values
            .get(legacy_output)
            .or_else(|| legacy_outputs.get(legacy_output));
        match (native_value, legacy_value) {
            (Some(native), Some(legacy)) => {
                cmp_field(
                    summary,
                    format!("node {idx} output value_info type"),
                    &native.type_info,
                    &legacy.type_info,
                );
                cmp_field(
                    summary,
                    format!("node {idx} output value_info exponents"),
                    &native.exponents,
                    &legacy.exponents,
                );
            }
            (None, None) => {}
            (None, Some(_)) => summary.fail(format!(
                "node {idx} native output {native_output:?} has no value_info while legacy output has one"
            )),
            (Some(_), None) => summary.fail(format!(
                "node {idx} legacy output {legacy_output:?} has no value_info while native output has one"
            )),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct ValueInfoSnap {
    name: Option<String>,
    type_info: Option<TypeInfoSnap>,
    exponents: Vec<i64>,
}

#[derive(Debug, PartialEq, Eq)]
struct TypeInfoSnap {
    value_type: String,
    denotation: Option<String>,
    tensor: Option<TensorTypeSnap>,
}

#[derive(Debug, PartialEq, Eq)]
struct TensorTypeSnap {
    elem_type: dl::TensorDataType,
    dims: Vec<DimensionSnap>,
}

#[derive(Debug, PartialEq, Eq)]
struct DimensionSnap {
    dim_type: String,
    dim_value: i64,
    dim_param: Option<String>,
    denotation: Option<String>,
}

fn value_info_vec(
    values: Option<flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<dl::ValueInfo<'_>>>>,
) -> Vec<ValueInfoSnap> {
    values
        .map(|values| {
            (0..values.len())
                .map(|i| {
                    let v = values.get(i);
                    ValueInfoSnap {
                        name: v.name().map(str::to_string),
                        type_info: v.value_info_type().map(type_info_snap),
                        exponents: i64_vec(v.exponents()),
                    }
                })
                .collect()
        })
        .unwrap_or_default()
}

fn value_info_map(
    values: Option<flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<dl::ValueInfo<'_>>>>,
) -> BTreeMap<String, ValueInfoSnap> {
    let mut out = BTreeMap::new();
    if let Some(values) = values {
        for i in 0..values.len() {
            let value = values.get(i);
            if let Some(name) = value.name() {
                out.insert(
                    name.to_string(),
                    ValueInfoSnap {
                        name: Some(name.to_string()),
                        type_info: value.value_info_type().map(type_info_snap),
                        exponents: i64_vec(value.exponents()),
                    },
                );
            }
        }
    }
    out
}

fn type_info_snap(t: dl::TypeInfo<'_>) -> TypeInfoSnap {
    TypeInfoSnap {
        value_type: format!("{:?}", t.value_type()),
        denotation: t.denotation().map(str::to_string),
        tensor: t.value_as_tensor_type().map(tensor_type_snap),
    }
}

fn tensor_type_snap(t: dl::TensorTypeAndShape<'_>) -> TensorTypeSnap {
    TensorTypeSnap {
        elem_type: t.elem_type(),
        dims: t
            .shape()
            .and_then(|shape| shape.dim())
            .map(|dims| {
                (0..dims.len())
                    .map(|i| {
                        let dim = dims.get(i);
                        let value = dim.value();
                        DimensionSnap {
                            dim_type: value
                                .map(|v| format!("{:?}", v.dim_type()))
                                .unwrap_or_default(),
                            dim_value: value.map(|v| v.dim_value()).unwrap_or_default(),
                            dim_param: value.and_then(|v| v.dim_param().map(str::to_string)),
                            denotation: dim.denotation().map(str::to_string),
                        }
                    })
                    .collect()
            })
            .unwrap_or_default(),
    }
}

fn compare_initializers(
    summary: &mut CompareSummary,
    native_nodes: &[NodeSnap],
    legacy_nodes: &[NodeSnap],
    native: BTreeMap<String, TensorSnap>,
    legacy: BTreeMap<String, TensorSnap>,
) {
    summary.initializer_count = native.len();
    if native.len() != legacy.len() {
        summary.fail(format!(
            "initializer count mismatch: native {} legacy {}",
            native.len(),
            legacy.len()
        ));
    }

    for (idx, (native_node, legacy_node)) in native_nodes.iter().zip(legacy_nodes).enumerate() {
        match native_node.op_type.as_deref() {
            Some("Conv") | Some("Gemm") => {
                compare_node_initializer(
                    summary,
                    &format!("node {idx} weight"),
                    native_node.inputs.get(1),
                    legacy_node.inputs.get(1),
                    &native,
                    &legacy,
                    true,
                );
                compare_node_initializer(
                    summary,
                    &format!("node {idx} bias"),
                    native_node.inputs.get(2),
                    legacy_node.inputs.get(2),
                    &native,
                    &legacy,
                    true,
                );
            }
            Some("ReduceMean") => {
                compare_node_initializer(
                    summary,
                    &format!("node {idx} axes"),
                    native_node.inputs.get(1),
                    legacy_node.inputs.get(1),
                    &native,
                    &legacy,
                    false,
                );
            }
            _ => {}
        }
    }
}

fn compare_node_initializer(
    summary: &mut CompareSummary,
    label: &str,
    native_name: Option<&String>,
    legacy_name: Option<&String>,
    native: &BTreeMap<String, TensorSnap>,
    legacy: &BTreeMap<String, TensorSnap>,
    tolerant_quant_payload: bool,
) {
    match (native_name, legacy_name) {
        (Some(native_name), Some(legacy_name)) => {
            let Some(native_tensor) = native.get(native_name) else {
                summary.fail(format!(
                    "{label}: native initializer {native_name:?} not found"
                ));
                return;
            };
            let Some(legacy_tensor) = legacy.get(legacy_name) else {
                summary.fail(format!(
                    "{label}: legacy initializer {legacy_name:?} not found"
                ));
                return;
            };
            compare_tensor(
                summary,
                label,
                native_tensor,
                legacy_tensor,
                tolerant_quant_payload,
            );
        }
        (None, None) => {}
        (None, Some(legacy_name)) => {
            summary.fail(format!(
                "{label}: native node has no initializer matching legacy {legacy_name:?}"
            ));
        }
        (Some(native_name), None) => {
            summary.fail(format!(
                "{label}: legacy node has no initializer matching native {native_name:?}"
            ));
        }
    }
}

fn compare_tensor(
    summary: &mut CompareSummary,
    label: &str,
    native: &TensorSnap,
    legacy: &TensorSnap,
    tolerant_quant_payload: bool,
) {
    cmp_field(
        summary,
        format!("{label} dtype"),
        native.data_type,
        legacy.data_type,
    );
    cmp_field(summary, format!("{label} dims"), &native.dims, &legacy.dims);
    cmp_field(
        summary,
        format!("{label} doc_string"),
        native.doc_string.as_ref(),
        legacy.doc_string.as_ref(),
    );
    cmp_field(
        summary,
        format!("{label} exponents"),
        &native.exponents,
        &legacy.exponents,
    );
    if tolerant_quant_payload {
        compare_quant_payload(summary, label, native, legacy);
    } else if native.raw_bytes != legacy.raw_bytes {
        summary.fail(format!(
            "{label}: raw payload mismatch (native {} bytes legacy {} bytes)",
            native.raw_bytes.len(),
            legacy.raw_bytes.len()
        ));
    }
}

#[derive(Debug)]
struct TensorSnap {
    data_type: dl::TensorDataType,
    dims: Vec<i64>,
    doc_string: Option<String>,
    exponents: Vec<i64>,
    raw_bytes: Vec<u8>,
    raw_values: Vec<i64>,
}

fn tensor_map(
    tensors: Option<flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<dl::Tensor<'_>>>>,
) -> BTreeMap<String, TensorSnap> {
    let mut out = BTreeMap::new();
    if let Some(tensors) = tensors {
        for i in 0..tensors.len() {
            let tensor = tensors.get(i);
            let name = tensor
                .name()
                .map(str::to_string)
                .unwrap_or_else(|| format!("<unnamed:{i}>"));
            out.insert(
                name,
                TensorSnap {
                    data_type: tensor.data_type(),
                    dims: i64_vec(tensor.dims()),
                    doc_string: tensor.doc_string().map(str::to_string),
                    exponents: i64_vec(tensor.exponents()),
                    raw_bytes: raw_bytes(&tensor),
                    raw_values: raw_int_values(&tensor),
                },
            );
        }
    }
    out
}

fn compare_quant_payload(
    summary: &mut CompareSummary,
    name: &str,
    native: &TensorSnap,
    legacy: &TensorSnap,
) {
    if native.raw_values.len() != legacy.raw_values.len() {
        summary.fail(format!(
            "{name}: quant payload length mismatch: native {} legacy {}",
            native.raw_values.len(),
            legacy.raw_values.len()
        ));
        return;
    }
    let mut bad = 0_usize;
    let mut max_abs_delta = 0_i64;
    let mut max_dequant_abs_delta = 0.0_f64;
    let mut examples = Vec::new();
    let native_scale = tensor_scale(native);
    let legacy_scale = tensor_scale(legacy);
    for (idx, (n, l)) in native.raw_values.iter().zip(&legacy.raw_values).enumerate() {
        let delta = (n - l).abs();
        max_abs_delta = max_abs_delta.max(delta);
        if let (Some(native_scale), Some(legacy_scale)) = (native_scale, legacy_scale) {
            let dequant_delta = ((*n as f64) * native_scale - (*l as f64) * legacy_scale).abs();
            max_dequant_abs_delta = max_dequant_abs_delta.max(dequant_delta);
        }
        if delta > 1 {
            bad += 1;
            if examples.len() < 5 {
                examples.push(format!("#{idx}: native {n} legacy {l}"));
            }
        }
    }
    let allowed = native.raw_values.len() / 100;
    summary.tensor_reports.push(format!(
        "{name}: compared {} values, >1 LSB diffs {bad}/{allowed}, max int delta {max_abs_delta}, max dequant abs delta {:.6e}",
        native.raw_values.len(),
        max_dequant_abs_delta,
    ));
    if native_scale.is_none() || legacy_scale.is_none() {
        summary.tensor_reports.push(format!(
            "{name}: dequant diff unavailable because one side has no single exponent"
        ));
    }
    if bad > allowed {
        summary.fail(format!(
            "{name}: {bad}/{} values differ by more than 1 LSB (allowed {allowed}); max dequant abs delta {:.6e}; examples: {}",
            native.raw_values.len(),
            max_dequant_abs_delta,
            examples.join(", ")
        ));
    }
}

fn tensor_scale(t: &TensorSnap) -> Option<f64> {
    let [exp] = t.exponents.as_slice() else {
        return None;
    };
    Some(2.0_f64.powi(*exp as i32))
}

fn raw_bytes(t: &dl::Tensor<'_>) -> Vec<u8> {
    let Some(raw) = t.raw_data() else {
        return Vec::new();
    };
    let mut bytes = Vec::with_capacity(raw.len() * 16);
    for i in 0..raw.len() {
        bytes.extend(raw.get(i).bytes().iter());
    }
    bytes
}

fn raw_int_values(t: &dl::Tensor<'_>) -> Vec<i64> {
    let dims_numel: usize = t
        .dims()
        .map(|v| (0..v.len()).map(|i| v.get(i) as usize).product())
        .unwrap_or(0);
    let bytes = raw_bytes(t);
    match t.data_type() {
        dl::TensorDataType::INT8 => bytes
            .into_iter()
            .take(dims_numel)
            .map(|b| (b as i8) as i64)
            .collect(),
        dl::TensorDataType::INT16 => bytes
            .chunks_exact(2)
            .take(dims_numel)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as i64)
            .collect(),
        dl::TensorDataType::INT32 => bytes
            .chunks_exact(4)
            .take(dims_numel)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
            .collect(),
        dl::TensorDataType::INT64 => bytes
            .chunks_exact(8)
            .take(dims_numel)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect(),
        _ => Vec::new(),
    }
}

fn str_vec(v: Option<flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<&str>>>) -> Vec<String> {
    v.map(|v| (0..v.len()).map(|i| v.get(i).to_string()).collect())
        .unwrap_or_default()
}

fn i64_vec(v: Option<flatbuffers::Vector<'_, i64>>) -> Vec<i64> {
    v.map(|v| (0..v.len()).map(|i| v.get(i)).collect())
        .unwrap_or_default()
}

fn cmp_field<T>(summary: &mut CompareSummary, label: impl AsRef<str>, native: T, legacy: T)
where
    T: PartialEq + std::fmt::Debug,
{
    if native != legacy {
        summary.fail(format!(
            "{} mismatch: native {native:?} legacy {legacy:?}",
            label.as_ref()
        ));
    }
}
