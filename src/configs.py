"""Every experimental configuration in the study. Letters follow the ablation list in the brief."""
R, P, H = "rgb", "pose", "hand"
ALL = [R, P, H]


def c(mods, fusion="concat", **kw):
    return dict(mods=list(mods), fusion=fusion, **kw)


CONFIGS = {
    # --- unimodal (A, B, C) and skeleton-only baseline
    "A_rgb": c([R]),
    "B_pose": c([P]),
    "C_hand": c([H]),
    "skel_concat": c([P, H]),
    # --- modality pairs (D, E, F) with fixed fusion and with the proposed fusion
    "D_rgb_pose_concat": c([R, P]),
    "E_rgb_hand_concat": c([R, H]),
    "F_pose_hand_concat": c([P, H]),
    "D_rgb_pose_trel": c([R, P], "trel_gate"),
    "E_rgb_hand_trel": c([R, H], "trel_gate"),
    "F_pose_hand_trel": c([P, H], "trel_gate"),
    # --- three modalities, fusion mechanism (G..L)
    "G_concat": c(ALL),                              # G/H: fixed fusion (concat + projection)
    "H_fixed_weight": c(ALL, "fixed_weight"),        # H: fixed scalar weights (input-independent)
    "I_seq_attention": c(ALL, "seq_attention"),      # I: attention fusion (per-sequence)
    "I_xattn": c(ALL, "xattn"),                      # I': cross-modal attention (MulT-style)
    "J_gate": c(ALL, "gate"),                        # J: adaptive per-frame gate from features
    "K_rel_gate": c(ALL, "rel_gate"),                # K: + observable quality signals
    "K_qual_gate": c(ALL, "qual_gate"),              # K': quality signals only
    "K_unc_gate": c(ALL, "unc_gate"),                # K'': + prediction uncertainty
    "K_rel_unc_gate": c(ALL, "rel_unc_gate"),
    "L_trel_gate": c(ALL, "trel_gate"),              # L: temporal reliability-aware (PROPOSED)
    "L_trel_unc_gate": c(ALL, "trel_unc_gate"),
    "twostream": c(ALL, "twostream"),                # strong contemporary multi-stream CSLR design
    "sentcls": c(ALL, "sentcls"),                    # diagnostic: closed-set sentence classifier
    # --- hand-appearance stream (DINOv2 on pose-located hand crops), added after Stage 1 showed the
    #     whole-frame embedding carries little sign information (see PREREGISTRATION amendment 2)
    "M_hrgb": c(["hrgb"]),
    "N_hrgb_hand_concat": c(["hrgb", H]),
    "N_hrgb_hand_trel": c(["hrgb", H], "trel_gate"),
    "Q_hrgb_pose_hand_concat": c(["hrgb", P, H]),
    "Q_hrgb_pose_hand_trel": c(["hrgb", P, H], "trel_gate"),
    "R_all4_concat": c([R, "hrgb", P, H]),
    "R_all4_trel": c([R, "hrgb", P, H], "trel_gate"),
    # --- controls
    "G_concat_d256": c(ALL, d=256),                  # parameter-matched/over-parameterised fixed fusion
    "G_concat_noaug": c(ALL, augment=False),
    "L_trel_gate_noaug": c(ALL, "trel_gate", augment=False),
    "G_concat_nomoddrop": c(ALL, mod_drop=0.0),
    "L_trel_gate_nomoddrop": c(ALL, "trel_gate", mod_drop=0.0),
    "G_concat_lstm": c(ALL, backbone="bilstm"),      # backbone ablation (BiLSTM failed to fit in pilots)
    "L_trel_gate_lstm": c(ALL, "trel_gate", backbone="bilstm"),
}

PROPOSED = "L_trel_gate"
# Candidate "strong baselines". The primary comparison uses whichever has the lowest mean
# *validation* WER on the primary protocol -- selected without looking at test results.
BASELINE_POOL = ["A_rgb", "skel_concat", "G_concat", "H_fixed_weight", "I_seq_attention", "I_xattn",
                 "twostream", "G_concat_d256"]
