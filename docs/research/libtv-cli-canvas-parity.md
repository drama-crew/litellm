# LibTV CLI canvas parity

The official LibTV CLI 1.1.3 uses different representations for canvas state and generation requests. A canvas video links resource nodes and stores rich references, reference ordering, `settings` and `advancedSettings`. The generation endpoint takes flat settings and model-specific asset references. Saving the wire request as canvas state lets the browser replace the requested mode, references and settings with defaults.

Implementation scope:

1. Serialize reference resource nodes, directed connections and ordered references for video generation. Keep CDN URLs and compliance asset IDs separate until wire serialization.
2. Use schema field names/defaults and `originalField` mappings for settings and advanced settings. Preserve explicit zero values.
3. Include the model in generation params and persist the task receipt/results on the originating canvas node. A canvas synchronization failure after paid submission must never cause a duplicate submission or lose the task ID.
4. Verify sync/async contracts, first/last ordering, mixed references, model aliases, canvas reload state and task recovery with targeted unit tests.
5. The account 1 Seedance 2.5 frames2video single-first-frame A/B completed: native CLI and the adapter prototype both failed with 16:9 and succeeded with adaptive, with identical remaining inputs (480p, 5 seconds, sound on, search off, automatic compliance on). The resulting LibTV canvas displays the reference and preserves the requested mode/settings.
6. Keep structured failure categories in progress parsing. INVALID_PARAMS must not trigger fresh-asset generation resubmissions just because its generic message says to retry; retain compatibility for the older uncategorized aging failure only.

Verification: 741 LibTV tests passed, one opt-in integration test skipped. Production deployment and Causyn public API/canvas acceptance follow main integration.

No unrelated production credential rotation is part of this change. Real tests use a temporary official account 1 CLI login in an isolated process. Preserve the existing failed API node and CLI draft as controls.
