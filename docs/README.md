# Project website and blog

This directory contains the project homepage and a Jekyll blog for GitHub Pages.
The existing `index.html` remains the homepage. Blog posts live in `_posts/`
and use the layout in `_layouts/post.html`.

## Publish later

1. Commit and push the docs changes to the branch you want to publish.
2. In the GitHub repository, open **Settings → Pages**.
3. Select **Deploy from a branch**, choose that branch, and select **/docs**.
4. Save and wait for the Pages deployment to finish.

The post will appear at `planning-together.html` relative to the website root.
Keep Jekyll enabled: do not add a `.nojekyll` file to this directory.

## Add another post

Create `_posts/YYYY-MM-DD-short-title.md` with front matter like this:

```yaml
---
layout: post
title: "Your post title"
description: "A short summary of the post."
date: YYYY-MM-DD 09:00:00 +0800
permalink: /short-title.html
---
```

Write the body in Markdown and add a link from `index.html` when ready.
Use relative asset paths such as `assets/ao_mrtamp.mp4` for these root-level
post URLs so links work under the repository's GitHub Pages subdirectory.

For a local preview with Jekyll installed, run `jekyll serve --source docs`
from the repository root. GitHub Pages performs the Jekyll build on deployment;
no generated HTML needs to be committed for Markdown posts.
