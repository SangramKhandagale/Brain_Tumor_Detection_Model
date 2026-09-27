"""
clinical_report.py
-------------------
Turns (predicted_class, confidence, all_probs, hotspot regions) into a
richly structured, radiology-report-style explanation modelled on real
neuroradiology reporting conventions.

Sections
--------
    STUDY INFORMATION
    AI MODEL SUMMARY
    IMPRESSION
    KEY FINDINGS  (hotspot-by-hotspot, with differential weight)
    REGION-LEVEL ANALYSIS
    RADIOLOGICAL CHARACTERISATION
    CONFIDENCE BREAKDOWN  (with calibration commentary)
    DIFFERENTIAL CONSIDERATIONS
    RECOMMENDED NEXT STEPS
    CLINICAL DISCLAIMER & LIMITATIONS

Design philosophy
-----------------
- Location is reported relative to the image frame (quadrant, laterality,
  distance from centre) rather than claimed anatomical lobe names, because
  a single 2-D slice without DICOM orientation metadata cannot reliably
  support lobe-level claims.
- Activation strength is expressed both as a raw percentage and a qualitative
  descriptor so the report is readable by both technical and clinical audiences.
- Class-specific differential reasoning surfaces the *why* behind the
  prediction, drawing on known imaging hallmarks of each tumour type.
- All text is deterministic (no randomness) — same inputs → same report —
  so the app remains reproducible and audit-friendly.
"""

from __future__ import annotations
import datetime

# ── CLASS KNOWLEDGE BASE ──────────────────────────────────────────────────────
# Each entry carries:
#   typical_appearance  – imaging hallmarks
#   common_locations    – anatomical origin (general, image-agnostic)
#   differential_keys   – features that help confirm OR exclude this diagnosis
#   grade_note          – grading / behaviour context
#   next_steps          – standard clinical follow-up pathway
#   icd_hint            – approximate ICD-10 code family for reference

CLASS_INFO: dict[str, dict] = {
    "Glioma": {
        "typical_appearance": (
            "Gliomas arise from glial cells (astrocytes, oligodendrocytes, ependymal cells) "
            "within the brain parenchyma. On conventional MRI they typically manifest as "
            "heterogeneous, infiltrative regions with indistinct margins. High-grade gliomas "
            "(WHO Grade III–IV, including glioblastoma) frequently show ring-like contrast "
            "enhancement, central necrosis, and perilesional vasogenic oedema on T2/FLAIR. "
            "Low-grade gliomas (WHO Grade I–II) are more homogeneous, hypointense on T1, "
            "and hyperintense on T2/FLAIR, with little or no enhancement."
        ),
        "common_locations": (
            "Most frequent in the cerebral hemispheres (frontal and temporal lobes in adults); "
            "brain-stem and cerebellar gliomas predominate in children."
        ),
        "differential_keys": {
            "supporting": [
                "Ill-defined, infiltrative margin crossing white-matter tracts",
                "T2/FLAIR hyperintensity extending beyond the enhancing core (oedema penumbra)",
                "Heterogeneous signal suggesting necrosis or haemorrhage",
                "Mass effect disproportionate to apparent lesion size",
            ],
            "against": [
                "Well-defined dural tail (more consistent with meningioma)",
                "Midline sella/suprasellar position (more consistent with pituitary adenoma)",
                "Extra-axial location with broad meningeal base",
            ],
        },
        "grade_note": (
            "Glioma grading under the 2021 WHO CNS Classification now integrates molecular "
            "markers (IDH mutation status, 1p/19q co-deletion, MGMT promoter methylation) "
            "alongside histology. Grade I–II: slower growth, better prognosis. "
            "Grade III–IV: aggressive, requiring urgent multidisciplinary management."
        ),
        "next_steps": [
            "Dedicated contrast-enhanced MRI with T1, T2, FLAIR, DWI/ADC, and SWI sequences",
            "MR spectroscopy (elevated Cho/Cr ratio, reduced NAA) and perfusion imaging (rCBV)",
            "Neurosurgical consultation for resection/biopsy planning",
            "Molecular profiling (IDH, EGFR, MGMT, 1p/19q) from tissue sample",
            "Multidisciplinary tumour board review",
        ],
        "icd_hint": "C71.x (Malignant neoplasm of brain — site-specific code required)",
    },

    "Meningioma": {
        "typical_appearance": (
            "Meningiomas arise from arachnoid cap cells of the meninges and are therefore "
            "extra-axial by definition. On MRI they are characteristically isointense to grey "
            "matter on T1, with intense and homogeneous contrast enhancement. The 'dural tail' "
            "sign — a thickening and enhancement of adjacent dura — is highly suggestive though "
            "not pathognomonic. Calcification is present in ~25 % of cases (hypodense on CT). "
            "They are typically well-circumscribed and displace, rather than infiltrate, the "
            "adjacent brain parenchyma."
        ),
        "common_locations": (
            "Parasagittal/falcine (25 %), convexity (20 %), sphenoid wing (20 %), "
            "olfactory groove, suprasellar, posterior fossa, and spinal canal."
        ),
        "differential_keys": {
            "supporting": [
                "Broad-based dural attachment with dural tail sign",
                "Homogeneous, intense contrast enhancement",
                "Extra-axial position (CSF cleft between mass and cortex on T2)",
                "Displacement rather than infiltration of adjacent brain",
                "Hyperdense on CT; possible calcification",
            ],
            "against": [
                "Intra-axial origin with parenchymal infiltration (favours glioma)",
                "Central necrosis or ring enhancement (favours high-grade glioma or abscess)",
                "Purely midline sellar position without dural component (favours pituitary lesion)",
            ],
        },
        "grade_note": (
            "~90 % of meningiomas are WHO Grade 1 (benign, slow-growing). WHO Grade 2 "
            "(atypical) and Grade 3 (anaplastic/malignant) account for ~10 % but carry "
            "higher recurrence risk. Brain invasion, defined histologically, upgrades a "
            "lesion to at least Grade 2 in the 2021 WHO scheme."
        ),
        "next_steps": [
            "Contrast-enhanced MRI with thin cuts through region of interest",
            "CT head for calcification characterisation and bone involvement",
            "Neurosurgical evaluation — management ranges from observation to microsurgical resection",
            "Stereotactic radiosurgery (SRS) consideration for small or residual lesions",
            "Ophthalmology or ENT referral if lesion abuts optic apparatus or skull base",
        ],
        "icd_hint": "D32.x (Benign neoplasm of meninges) or C70.x if malignant",
    },

    "Pituitary": {
        "typical_appearance": (
            "Pituitary adenomas arise from the anterior pituitary (adenohypophysis) within the "
            "sella turcica. Microadenomas (< 10 mm) appear as focal hypointense lesions within "
            "the normally enhancing gland on T1 post-contrast; they may cause subtle gland "
            "asymmetry or stalk deviation. Macroadenomas (≥ 10 mm) extend superiorly to "
            "compress the optic chiasm (classically producing bitemporal hemianopia), inferiorly "
            "through the sellar floor, or laterally to invade the cavernous sinus. "
            "Haemorrhage into the tumour (pituitary apoplexy) produces heterogeneous T1 signal."
        ),
        "common_locations": (
            "Sella turcica and parasellar region; suprasellar extension in macroadenomas. "
            "Cavernous sinus invasion indicates locally aggressive behaviour."
        ),
        "differential_keys": {
            "supporting": [
                "Central midline sellar/suprasellar mass",
                "Expansion or remodelling of the sella turcica on CT",
                "Stalk deviation or gland asymmetry (microadenoma)",
                "Figure-of-eight or 'snowman' suprasellar extension (macroadenoma)",
                "Rim enhancement with central hypo-intensity (Rathke cleft cyst DDx)",
            ],
            "against": [
                "Peripheral, extra-axial dural-based mass (favours meningioma)",
                "Infiltrative parenchymal involvement without sellar epicentre (favours glioma)",
                "Purely intra-axial location well above the sella",
            ],
        },
        "grade_note": (
            "Pituitary tumours are now classified as pituitary neuroendocrine tumours (PitNETs). "
            "Functioning adenomas (secreting GH, prolactin, ACTH, TSH) cause systemic hormonal "
            "syndromes (acromegaly, hyperprolactinaemia, Cushing's disease). Non-functioning "
            "adenomas present through mass effect. Aggressive or invasive behaviour does not "
            "reliably correlate with histological features alone."
        ),
        "next_steps": [
            "Dedicated sella MRI protocol: thin-slice (2–3 mm) T1 pre- and post-Gd, T2, coronal/sagittal",
            "Full pituitary hormone panel: IGF-1, prolactin, ACTH, cortisol, TSH, LH/FSH",
            "Formal visual-field perimetry (Goldman/Humphrey) if suprasellar extension",
            "Endocrinology referral for hormonal management",
            "Neurosurgical consultation for trans-sphenoidal resection if indicated",
        ],
        "icd_hint": "D35.2 (Benign neoplasm of pituitary gland) or E22–E23.x for functional variants",
    },

    "No Tumor": {
        "typical_appearance": (
            "No region of abnormal signal intensity or mass-like enhancement was identified "
            "as a dominant focus of model attention. The AI activation map did not concentrate "
            "on any sub-region above the reporting threshold."
        ),
        "common_locations": "N/A — negative screening result.",
        "differential_keys": {
            "supporting": [
                "Diffuse, low-level activation without spatial concentration",
                "No quadrant or lateralised focus above threshold",
            ],
            "against": [
                "A confident focal hotspot in any quadrant would argue against this class",
            ],
        },
        "grade_note": (
            "A negative AI screen reduces — but does not eliminate — the probability of a "
            "detectable intracranial mass on this sequence and slice. Subtle low-grade lesions, "
            "lesions at slice edges, or pathology outside this model's training distribution "
            "may not be detected."
        ),
        "next_steps": [
            "Clinical correlation with patient symptoms and neurological examination",
            "If clinical suspicion remains, full MRI brain with contrast and multi-sequence protocol",
            "Routine follow-up as directed by the treating clinician",
        ],
        "icd_hint": "Z03.89 (Encounter for observation for other suspected diseases — ruled out)",
    },
}


# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────

def _quadrant_label(cx: float, cy: float) -> str:
    """
    Convert normalised (0–1) centroid to a precise image-relative description.
    Also computes an approximate distance from image centre for extra specificity.
    """
    # Vertical thirds
    if cy < 0.33:
        vertical = "upper third"
    elif cy < 0.67:
        vertical = "middle third"
    else:
        vertical = "lower third"

    # Horizontal thirds
    if cx < 0.33:
        horizontal = "left"
    elif cx < 0.67:
        horizontal = "central"
    else:
        horizontal = "right"

    # Laterality qualifier
    if cx < 0.45:
        laterality = "left-lateralised"
    elif cx > 0.55:
        laterality = "right-lateralised"
    else:
        laterality = "midline-aligned"

    # Distance from centre (0 = centre, 0.707 = corner)
    dist = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5
    if dist < 0.15:
        proximity = "close to the image centre"
    elif dist < 0.35:
        proximity = "paracentral"
    else:
        proximity = "peripheral/eccentric"

    if horizontal == "central" and vertical == "middle third":
        return "central region of the image (near isocentre)"
    return (
        f"{vertical}, {horizontal} sector — {laterality}, {proximity}"
    )


def _size_descriptor(area_frac: float) -> tuple[str, str]:
    """Return (qualitative label, clinical implication) for the hotspot extent."""
    if area_frac < 0.01:
        return "punctate / focal", "consistent with a small, well-circumscribed lesion or early lesion"
    elif area_frac < 0.03:
        return "small", "may represent a microlesion; borderline for spatial interpretability"
    elif area_frac < 0.08:
        return "moderate", "occupies a meaningful sub-region; spatially interpretable"
    elif area_frac < 0.18:
        return "large", "substantial regional involvement; consider mass effect"
    else:
        return "extensive", (
            "widespread activation — may reflect a large lesion, significant oedema, "
            "or diffuse infiltration; alternatively could indicate model uncertainty"
        )


def _activation_descriptor(peak: float) -> tuple[str, str]:
    """Return (qualitative label, interpretation note) for peak Grad-CAM activation."""
    if peak >= 0.85:
        return "very strong", "exceptionally high model certainty in this spatial region"
    elif peak >= 0.65:
        return "strong", "high localisation confidence — region likely core to the prediction"
    elif peak >= 0.45:
        return "moderate", "meaningful but not dominant driver of the classification"
    elif peak >= 0.25:
        return "weak", "marginal contribution; treat spatial location with caution"
    else:
        return "minimal", "below typical significance threshold — low spatial confidence"


def _confidence_descriptor(confidence: float) -> tuple[str, str]:
    """Return (label, clinical weight) for the overall softmax confidence."""
    if confidence >= 0.95:
        return "very high", "Model output is highly decisive; probability mass strongly concentrated on this class."
    elif confidence >= 0.80:
        return "high", "Output probability clearly favours this class over all alternatives."
    elif confidence >= 0.60:
        return "moderate", "Prediction is leading but not conclusive; consider differential classes."
    elif confidence >= 0.40:
        return "low-moderate", "Two or more classes score comparably — interpret with caution."
    else:
        return "low", (
            "Model is uncertain across classes. This output should be treated as "
            "non-diagnostic and reviewed by a specialist."
        )


def _entropy_note(all_probs) -> str:
    """Compute Shannon entropy of the probability distribution and interpret it."""
    import math
    eps = 1e-9
    H = -sum(p * math.log2(p + eps) for p in all_probs)
    n = len(all_probs)
    max_H = math.log2(n)
    norm_H = H / max_H if max_H > 0 else 0

    if norm_H < 0.2:
        return (
            f"Prediction entropy: **{H:.2f} bits** (normalised {norm_H:.0%}) — "
            "distribution is sharply peaked; the model is confident in its top class."
        )
    elif norm_H < 0.5:
        return (
            f"Prediction entropy: **{H:.2f} bits** (normalised {norm_H:.0%}) — "
            "moderate spread across classes; consider the top two differentials together."
        )
    else:
        return (
            f"Prediction entropy: **{H:.2f} bits** (normalised {norm_H:.0%}) — "
            "high uncertainty; probability mass is widely distributed. "
            "This scan may be atypical or outside the model's confident operating range."
        )


def _dominance_ratio(top_prob: float, second_prob: float) -> str:
    if second_prob < 1e-6:
        return "No meaningful runner-up class."
    ratio = top_prob / second_prob
    if ratio >= 5:
        return f"Top class is {ratio:.1f}× more probable than the runner-up — high class separation."
    elif ratio >= 2:
        return f"Top class is {ratio:.1f}× more probable than the runner-up — clear but not dominant lead."
    else:
        return (
            f"Top class is only {ratio:.1f}× more probable than the runner-up — "
            "differential diagnosis cannot be dismissed."
        )


# ── MAIN REPORT GENERATOR ─────────────────────────────────────────────────────

def generate_clinical_report(
    predicted_class: str,
    confidence: float,
    all_probs,          # array-like of float, ordered to match class_names
    class_names,        # list of str
    regions: list,      # list of dicts with keys: centroid_norm, area_frac, peak_activation
) -> str:
    """
    Build the full markdown report string.

    Parameters
    ----------
    predicted_class : str
        Top-1 predicted class name.
    confidence : float
        Softmax probability for the top-1 class (0–1).
    all_probs : sequence of float
        Full softmax probability vector, ordered as class_names.
    class_names : sequence of str
        Ordered list of class labels matching all_probs.
    regions : list of dict
        Grad-CAM hotspot descriptors. Each dict should contain:
            centroid_norm   : (float, float) — (cx, cy) in [0,1]×[0,1]
            area_frac       : float — fraction of image area covered by hotspot
            peak_activation : float — maximum normalised Grad-CAM value in region
    """
    info = CLASS_INFO.get(predicted_class, CLASS_INFO["No Tumor"])
    conf_label, conf_weight = _confidence_descriptor(confidence)
    all_probs = list(all_probs)

    # Sort classes by probability for downstream use
    ranked = sorted(zip(class_names, all_probs), key=lambda x: -x[1])
    top_name, top_prob = ranked[0]
    second_name, second_prob = ranked[1] if len(ranked) > 1 else ("—", 0.0)

    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []

    # ═══════════════════════════════════════════════════════════════════════════
    # HEADER / STUDY METADATA
    # ═══════════════════════════════════════════════════════════════════════════
    lines.append("---")
    lines.append("## 🧠 AI-Assisted Neuroradiology Screening Report")
    lines.append(f"> **Generated:** {timestamp}  \n"
                 "> **Modality:** MRI Brain (single 2-D slice, plane unspecified)  \n"
                 "> **AI Model:** CNN Classifier + Grad-CAM Explainability  \n"
                 "> **Report Status:** *DRAFT — for review by qualified clinician only*")
    lines.append("\n---")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — AI MODEL SUMMARY (executive one-liner)
    # ═══════════════════════════════════════════════════════════════════════════
    lines.append("\n## § 1 · AI Model Summary")
    lines.append(
        f"| Parameter | Value |\n"
        f"|-----------|-------|\n"
        f"| **Predicted Class** | {predicted_class} |\n"
        f"| **Softmax Confidence** | {confidence:.1%} ({conf_label}) |\n"
        f"| **Number of Attention Hotspots** | {len(regions)} |\n"
        f"| **Runner-Up Class** | {second_name} ({second_prob:.1%}) |"
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — IMPRESSION
    # ═══════════════════════════════════════════════════════════════════════════
    lines.append("\n---\n## § 2 · Impression")

    if predicted_class == "No Tumor":
        lines.append(
            f"The convolutional neural network classified this MRI slice as **No Tumor** "
            f"with **{conf_label} confidence ({confidence:.1%})**. {conf_weight} "
            f"The spatial activation map (Grad-CAM) did not produce any focal hotspot "
            f"above the reporting threshold, indicating that no single image sub-region "
            f"dominated the model's decision — a pattern consistent with a negative screen.\n\n"
            f"{info['grade_note']}"
        )
    else:
        n_regions = len(regions)
        region_word = "region" if n_regions == 1 else "regions"
        lines.append(
            f"The convolutional neural network classified this MRI slice as "
            f"**{predicted_class}** with **{conf_label} confidence ({confidence:.1%})**. "
            f"{conf_weight}\n\n"
            f"The Grad-CAM explainability map identified **{n_regions} attention "
            f"{region_word}** that most strongly contributed to this classification. "
            f"In general, {info['typical_appearance']}\n\n"
            f"**Typical anatomical origin:** {info['common_locations']}"
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — KEY FINDINGS (hotspot-level)
    # ═══════════════════════════════════════════════════════════════════════════
    lines.append("\n---\n## § 3 · Key Findings — Model Attention Hotspots")

    if not regions or predicted_class == "No Tumor":
        lines.append(
            "No spatially concentrated activation region was identified above threshold. "
            "This is consistent with the negative classification, though diffuse low-level "
            "activations scattered across the image are not individually reportable."
        )
    else:
        for i, r in enumerate(regions, start=1):
            cx, cy = r["centroid_norm"]
            location = _quadrant_label(cx, cy)
            size_label, size_note = _size_descriptor(r["area_frac"])
            act_label, act_note = _activation_descriptor(r["peak_activation"])

            lines.append(f"\n### Hotspot #{i}")
            lines.append(
                f"| Attribute | Detail |\n"
                f"|-----------|--------|\n"
                f"| **Image Location** | {location} |\n"
                f"| **Centroid (normalised)** | x = {cx:.3f}, y = {cy:.3f} |\n"
                f"| **Spatial Extent** | {size_label} (~{r['area_frac']*100:.1f}% of image area) |\n"
                f"| **Peak Activation** | {r['peak_activation']*100:.0f}% — {act_label} |\n"
                f"| **Activation Interpretation** | {act_note} |"
            )
            lines.append(
                f"\n**Size interpretation:** {size_note}.\n\n"
                f"This region is the {'primary' if i == 1 else 'secondary'} spatial driver "
                f"of the **{predicted_class}** prediction. "
                f"{'The model attended most strongly here, making it the most diagnostically relevant location in this scan.' if i == 1 else 'Its contribution supplements the primary hotspot; consider both jointly.'}"
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — RADIOLOGICAL CHARACTERISATION
    # ═══════════════════════════════════════════════════════════════════════════
    lines.append("\n---\n## § 4 · Radiological Characterisation")
    lines.append(
        "The following summarises known imaging hallmarks for the predicted class. "
        "These are **reference patterns from literature** — a formal assessment requires "
        "multi-sequence MRI with contrast and expert radiological review.\n"
    )

    lines.append(f"**Typical MRI Appearance — {predicted_class}**\n")
    lines.append(info["typical_appearance"])

    if predicted_class != "No Tumor":
        lines.append("\n**Features Supporting This Classification:**")
        for feat in info["differential_keys"]["supporting"]:
            lines.append(f"- ✅ {feat}")

        lines.append("\n**Features That Would Argue Against This Classification:**")
        for feat in info["differential_keys"]["against"]:
            lines.append(f"- ❌ {feat}")

        lines.append(f"\n**Grading & Behaviour Note:**\n{info['grade_note']}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — CONFIDENCE BREAKDOWN
    # ═══════════════════════════════════════════════════════════════════════════
    lines.append("\n---\n## § 5 · Confidence Breakdown")
    lines.append("Softmax probability distribution across all classes:\n")

    # Bar chart using Unicode block characters (renders in monospace-capable viewers)
    BAR_WIDTH = 30
    for cname, prob in ranked:
        filled = round(prob * BAR_WIDTH)
        bar = "█" * filled + "░" * (BAR_WIDTH - filled)
        marker = " ◄ **TOP**" if cname == predicted_class else ""
        lines.append(f"- **{cname}**: `{bar}` {prob:.1%}{marker}")

    lines.append(f"\n{_entropy_note(all_probs)}")
    lines.append(f"\n{_dominance_ratio(top_prob, second_prob)}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 6 — DIFFERENTIAL CONSIDERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    lines.append("\n---\n## § 6 · Differential Diagnostic Considerations")

    if predicted_class != "No Tumor":
        runner_up_info = CLASS_INFO.get(second_name)
        lines.append(
            f"Given a confidence of {confidence:.1%}, the following differential "
            f"should be kept in mind alongside the primary prediction:\n"
        )

        for rank_idx, (cname, prob) in enumerate(ranked):
            if cname == predicted_class:
                weight = "**Primary prediction — highest probability**"
            elif prob >= 0.15:
                weight = "**Significant differential — cannot be excluded without further imaging**"
            elif prob >= 0.05:
                weight = "Possible differential — lower probability but warrants consideration"
            else:
                weight = "Unlikely — low model probability"

            lines.append(f"- **{cname}** ({prob:.1%}): {weight}")

        if runner_up_info and second_prob >= 0.10:
            lines.append(
                f"\n> 📌 **Note on {second_name} ({second_prob:.1%}):** "
                f"{runner_up_info['typical_appearance'][:200]}…"
            )
    else:
        lines.append(
            "Negative classification. If clinical suspicion for an intracranial mass "
            "is present despite this negative AI screen, the following diagnoses "
            "should be considered on full clinical MRI:\n"
            "- Low-grade glioma (may be subtle on single sequence)\n"
            "- Cortical dysplasia or developmental anomaly\n"
            "- Early or small meningioma at image periphery\n"
            "- Non-neoplastic pathology: demyelination, infarct, abscess"
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 7 — RECOMMENDED NEXT STEPS
    # ═══════════════════════════════════════════════════════════════════════════
    lines.append("\n---\n## § 7 · Recommended Next Steps")
    lines.append(
        "The following pathway reflects standard clinical practice for this "
        "classification. **All recommendations are subject to clinical judgement "
        "and must be validated by the responsible clinician.**\n"
    )
    for step_num, step in enumerate(info["next_steps"], start=1):
        lines.append(f"{step_num}. {step}")

    lines.append(f"\n> **Approximate ICD-10 Reference:** {info['icd_hint']}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 8 — LIMITATIONS & DISCLAIMER
    # ═══════════════════════════════════════════════════════════════════════════
    lines.append("\n---\n## § 8 · Limitations & Disclaimer")
    lines.append(
        "**Spatial Interpretation Constraints**\n"
        "- All hotspot locations are expressed **relative to the image frame** "
        "(e.g., 'upper-left sector'), not mapped to anatomical lobe names. "
        "A single 2-D MRI slice without DICOM orientation metadata (axial / sagittal / "
        "coronal) cannot reliably support lobe-level claims (e.g., 'frontal lobe') "
        "without risk of systematic mislabelling.\n"
        "- Grad-CAM visualises *where the model attended*, which is not identical "
        "to the true lesion boundary. Activation can be driven by surrounding "
        "oedema, contrast artefact, or skull/tissue interfaces that correlate "
        "with but are not the lesion itself.\n"
        "- Hotspot area fraction is an approximation of activation coverage, "
        "not a volumetric tumour measurement.\n\n"
        "**Model Scope Constraints**\n"
        "- This model was trained on a limited dataset of four classes. "
        "Pathologies outside this set (e.g., abscesses, vascular malformations, "
        "metastases, demyelinating plaques) may be misclassified.\n"
        "- Performance may degrade on images from different MRI scanners, field "
        "strengths, or acquisition protocols not represented in training data.\n"
        "- A single 2-D slice lacks the volumetric context provided by a full "
        "3-D MRI series; lesion characterisation from one slice is inherently limited.\n\n"
        "**Clinical Disclaimer**\n"
        "> ⚠️ **This report is generated by an AI model and constitutes a screening "
        "aid only. It is NOT a radiological diagnosis, NOT a substitute for expert "
        "clinical assessment, and must NOT be used as the sole basis for any "
        "clinical decision. All findings require confirmation by a qualified "
        "radiologist or treating physician using the full clinical imaging "
        "context, patient history, and appropriate follow-up investigations.**"
    )

    lines.append("\n---\n*End of AI-assisted screening report.*")

    return "\n".join(lines)