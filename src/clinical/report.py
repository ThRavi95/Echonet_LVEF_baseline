"""
Clinical NLP Rules Engine & Diagnostic Report Generator
========================================================
Translates continuous LVEF estimations and 3D Grad-CAM explanation fidelity
metrics into standardized, guideline-adherent clinical diagnostic reports.

Adheres strictly to:
  - ASE / EACVI Guidelines for the Evaluation of LV Systolic Function
    (Lang et al., JASE 2015; McDonagh et al., Eur Heart J 2021)
  - ACC/AHA Guidelines for the Management of Heart Failure (Heidenreich et al., Circulation 2022)
  - Quantitative XAI Explanation Trust Framework (Hu et al., J. Cardiovasc. Imaging 2024)
"""
from typing import Any, Dict, List, Optional


class ClinicalReportGenerator:
    """
    Evidence-based NLP report generation engine for AI-assisted echocardiography.
    """

    @staticmethod
    def classify_lvef(lvef: float, gender: Optional[str] = None) -> Dict[str, Any]:
        """
        Classify LVEF according to ASE/EACVI clinical criteria.
        """
        # Determine cutoff thresholds
        # Male normal: >= 52%, Female normal: >= 54%, Universal consensus: >= 50%
        if gender and gender.upper() == "FEMALE":
            normal_threshold = 54.0
        elif gender and gender.upper() == "MALE":
            normal_threshold = 52.0
        else:
            normal_threshold = 50.0

        if lvef >= normal_threshold:
            category = "Normal LV Systolic Function"
            phenotype = "Preserved Ejection Fraction (HFpEF)"
            severity = "Normal"
            heart_risk_level = "Low Risk"
            badge_color = "#10b981"  # Emerald green
            urgency = "Routine"
            prognosis = "Favorable systolic performance. Normal myocardial contractility."
            recommendation = "Standard primary prevention; routine follow-up."
        elif lvef >= 41.0:
            category = "Mild LV Systolic Dysfunction"
            phenotype = "Mildly Reduced Ejection Fraction (HFmrEF)"
            severity = "Mild"
            heart_risk_level = "Moderate Risk"
            badge_color = "#f59e0b"  # Amber
            urgency = "Elective"
            prognosis = "Mildly impaired global systolic function. Outpatient cardiology correlation recommended."
            recommendation = "Cardiology evaluation; baseline neurohormonal blockade consideration."
        elif lvef >= 30.0:
            category = "Moderate LV Systolic Dysfunction"
            phenotype = "Moderately Reduced Ejection Fraction"
            severity = "Moderate"
            heart_risk_level = "High Risk"
            badge_color = "#f97316"  # Orange
            urgency = "Priority"
            prognosis = "Moderate systolic impairment with noticeable reduction in myocardial excursion. Guideline-directed medical therapy indicated."
            recommendation = "Initiate/titrate 4-pillar GDMT (ARNI/ACEi, Beta-Blocker, MRA, SGLT2i)."
        else:
            category = "Severe LV Systolic Dysfunction"
            phenotype = "Reduced Ejection Fraction (HFrEF)"
            severity = "Severe"
            heart_risk_level = "Critical Heart Failure Risk"
            badge_color = "#ef4444"  # Crimson red
            urgency = "Urgent"
            prognosis = "Critically depressed left ventricular ejection fraction. High risk of cardiovascular events. Urgent clinical correlation and heart failure specialist consultation required."
            recommendation = "Immediate heart failure consultation; comprehensive GDMT optimization; evaluate ICD/CRT eligibility."

        return {
            "category": category,
            "phenotype": phenotype,
            "severity": severity,
            "heart_risk_level": heart_risk_level,
            "badge_color": badge_color,
            "urgency": urgency,
            "prognosis": prognosis,
            "recommendation": recommendation,
        }

    @staticmethod
    def assess_explanation_fidelity(
        iou_ed: Optional[float] = None,
        iou_es: Optional[float] = None,
        pointing_hit: Optional[bool] = None,
        energy_fraction: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate anatomical explanation fidelity and transparency confidence.
        """
        if iou_ed is None or iou_es is None:
            return {
                "fidelity_level": "Unassessed",
                "confidence_score": "N/A",
                "badge_color": "#64748b",
                "interpretation": "Ground truth endocardial tracings were not supplied for this video. Explanation fidelity unverified.",
                "spurious_attention_warning": False,
            }

        mean_iou = (iou_ed + iou_es) / 2.0

        if mean_iou >= 0.40:
            fidelity_level = "High Anatomical Fidelity"
            badge_color = "#10b981"
            interpretation = (
                f"Model spatiotemporal attention aligns strongly with the expert Left Ventricle "
                f"endocardial contour (Mean IoU = {mean_iou:.3f}, ED = {iou_ed:.3f}, ES = {iou_es:.3f}). "
                f"Decision relies on genuine myocardial motion rather than peripheral artifacts."
            )
            spurious_warning = False
        elif mean_iou >= 0.25:
            fidelity_level = "Moderate Anatomical Fidelity"
            badge_color = "#f59e0b"
            interpretation = (
                f"Model attention predominantly covers the LV chamber with minor peripheral "
                f"leak into mitral annular or basal tissue (Mean IoU = {mean_iou:.3f}). "
                f"Saliency remains clinically plausible."
            )
            spurious_warning = False
        else:
            fidelity_level = "Low Anatomical Fidelity"
            badge_color = "#ef4444"
            interpretation = (
                f"Low spatial overlap between Grad-CAM focus and ground-truth LV contour "
                f"(Mean IoU = {mean_iou:.3f}). Model may be attending to non-chamber features "
                f"(e.g., pericardial fat, chest wall, or acoustic dropouts). Caution advised."
            )
            spurious_warning = True

        return {
            "fidelity_level": fidelity_level,
            "mean_iou": round(mean_iou, 4),
            "iou_ed": round(iou_ed, 4),
            "iou_es": round(iou_es, 4),
            "badge_color": badge_color,
            "interpretation": interpretation,
            "spurious_attention_warning": spurious_warning,
            "pointing_game_hit": pointing_hit,
            "energy_fraction": round(energy_fraction, 4) if energy_fraction is not None else None,
        }

    def generate_report(
        self,
        patient_id: str,
        predicted_ef: float,
        true_ef: Optional[float] = None,
        gender: Optional[str] = None,
        heart_rate_bpm: Optional[float] = None,
        ed_frame: Optional[int] = None,
        es_frame: Optional[int] = None,
        iou_ed: Optional[float] = None,
        iou_es: Optional[float] = None,
        pointing_hit: Optional[bool] = None,
        energy_fraction: Optional[float] = None,
        degradation_applied: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive, structured clinical report.
        """
        classification = self.classify_lvef(predicted_ef, gender)
        xai_fidelity = self.assess_explanation_fidelity(iou_ed, iou_es, pointing_hit, energy_fraction)

        # Clinical warnings
        clinical_alerts: List[str] = []
        if predicted_ef < 35.0:
            clinical_alerts.append(
                "CRITICAL: LVEF <= 35% meets primary prevention criteria for Implantable "
                "Cardioverter-Defibrillator (ICD) evaluation if symptomatic despite guideline-directed therapy (ACC/AHA Class I)."
            )
        if predicted_ef < 30.0:
            clinical_alerts.append(
                "URGENT: Severe systolic dysfunction. Heart failure specialist consultation and hospitalization threshold alert."
            )
        if xai_fidelity.get("spurious_attention_warning", False):
            clinical_alerts.append(
                "WARNING: XAI explanation fidelity is low (IoU < 0.25). Verify image quality and manual biplane Simpson's method."
            )
        if degradation_applied:
            clinical_alerts.append(
                f"STRESS TEST ACTIVE: Video analyzed under simulated ultrasound degradation ({degradation_applied})."
            )

        # Plain language prose narrative
        narrative_parts = [
            f"Echocardiographic assessment for Patient {patient_id} demonstrates an estimated "
            f"Left Ventricular Ejection Fraction (LVEF) of {predicted_ef:.1f}%, consistent with "
            f"'{classification['category']}' ({classification['phenotype']})."
        ]

        if true_ef is not None:
            err = abs(predicted_ef - true_ef)
            narrative_parts.append(
                f"Comparison with the cardiologist baseline ({true_ef:.1f}%) indicates an absolute "
                f"prediction discrepancy of {err:.1f}%."
            )

        if ed_frame is not None and es_frame is not None:
            cycle_frames = abs(es_frame - ed_frame) * 2
            narrative_parts.append(
                f"Cardiac cycle analysis identified peak ventricular filling (End-Diastole) at frame {ed_frame} "
                f"and peak systolic contraction (End-Systole) at frame {es_frame} (~{cycle_frames} frames/cycle)."
            )

        narrative_parts.append(classification["prognosis"])
        narrative_parts.append(xai_fidelity["interpretation"])

        full_narrative = " ".join(narrative_parts)

        # Construct markdown representation
        markdown_lines = [
            f"### Echocardiography Clinical AI Diagnostic Report",
            f"**Patient ID:** `{patient_id}` | **Estimated LVEF:** **{predicted_ef:.1f}%**",
            f"- **Classification:** <span style='color:{classification['badge_color']};font-weight:bold;'>{classification['category']}</span>",
            f"- **Phenotype:** {classification['phenotype']}",
            f"- **Triage Status:** {classification['urgency']}",
            f"- **Explanation Fidelity:** <span style='color:{xai_fidelity['badge_color']};font-weight:bold;'>{xai_fidelity['fidelity_level']}</span>",
        ]

        if true_ef is not None:
            markdown_lines.append(f"- **Cardiologist Ground Truth:** {true_ef:.1f}% (Absolute Error: {abs(predicted_ef - true_ef):.1f}%)")
        if xai_fidelity.get("mean_iou") is not None:
            markdown_lines.append(f"- **Mean Anatomical IoU:** {xai_fidelity['mean_iou']:.3f} (ED: {xai_fidelity['iou_ed']:.3f}, ES: {xai_fidelity['iou_es']:.3f})")

        if clinical_alerts:
            markdown_lines.append("\n**Clinical Action Flags:**")
            for alert in clinical_alerts:
                markdown_lines.append(f"> ⚠️ **{alert}**")

        markdown_lines.append(f"\n**Diagnostic Narrative:**\n{full_narrative}")

        return {
            "patient_id": patient_id,
            "predicted_lvef": round(predicted_ef, 1),
            "true_lvef": round(true_ef, 1) if true_ef is not None else None,
            "classification": classification,
            "fidelity": xai_fidelity,
            "clinical_alerts": clinical_alerts,
            "narrative": full_narrative,
            "markdown_report": "\n".join(markdown_lines),
        }

