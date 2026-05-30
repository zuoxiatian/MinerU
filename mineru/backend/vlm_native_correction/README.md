# VLM Native Correction Backend

`vlm_native_correction` is a VLM-primary backend:

- VLM owns layout, block order, visual structure, whitespace, and line breaks.
- PDF native text is only used as a per-bbox correction source for ordinary text.
- Structural blocks such as tables, images, equations, and code are kept from VLM.

The backend is intended for documents where native PDF text is useful for character
accuracy, but native ordering, line breaks, and formatting are unreliable.

