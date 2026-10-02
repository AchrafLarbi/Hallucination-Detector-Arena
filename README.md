---
title: Hallucination Detector Arena
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.50.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Can we trust LLM hallucination detectors?
startup_duration_timeout: 1h
---

<!-- The block above is configuration for Hugging Face Spaces (GitHub shows it as a table). -->

# Hallucination Detector Arena

**Can we trust LLM hallucination detectors?** Teams increasingly use automatic detectors to measure how often
their LLM "hallucinates", but a hallucination rate is only as valid as the detector that produced it. This project
audits four detectors on answers whose truth is known, across six domains, with a protocol designed to avoid
inflated numbers, and serves the results and a live playground as a web app.

**Author:** [Mohammed Achraf Larbi](https://github.com/AchrafLarbi)
&nbsp;·&nbsp; **Live app:** [huggingface.co/spaces/achraf2203/hallucination-arena](https://huggingface.co/spaces/achraf2203/hallucination-arena)
