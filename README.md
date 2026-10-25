# All things ones

Experiments in making sets of images that hide a picture. Each image is printed on its own clear acrylic sheet (e.g. a coaster). On its own, each sheet should look like an unrelated picture or pattern. Stacked together, the sheets reveal a hidden target image.

Inspired by [All and One](https://www.artof01.com/vrellis/works/AllAndOne.html).

There are two ways the sheets can combine, depending on how they're printed:

- **Multiply light**: translucent ink, like coloured film. Light passes through every sheet, so the stack's colour is the product of every layer (0 = black, 1 = clear). Stacking can only darken.
- **Transparency**: opaque ink, like stickers. You see the top-most painted pixel, so upper layers hide lower ones.

## Running

```sh
mise install
mise ip
mise script shatter
```

Run scripts from the repo root. Settings are at the top of each script, and each step is a function called from `main()` so steps can be commented out while experimenting. Outputs go to `data/output/<script>/`.

## Scripts

### Multiply light

Layers are printed on transparent film, so the stacked colour is the product of each layer.

**shatter.py** - shatter the target into pieces dealt across the layers, turn each layer's pieces into a colourful AI pattern (manual step, saved to `data/input/seed/<name>/shattered_piece_XX_gen_<suffix>.jpg`), then nudge the patterns until the stack reveals the target. Outputs are numbered by step.

**pyramid.py** - build the layers from coarse to fine detail. For each blur level, find blobs where the stack doesn't match the blurred goal, paint them (plus decoys) onto the layers, then correct the stack. Saves each blur level to `pass_NN_blur_XX/`, with files numbered by step.

**blob_levels.py** - split the target into blobs and give each blob a light random colour on every layer. Add turned and flipped copies of the target's detail as decoys. Then slowly adjust the layers until the stack matches the target, giving each adjustment to a random layer so the picture is spread thinly across them all. Prints a score for how much of the target shows in each layer. Outputs are numbered by step.

### Transparency

Layers are opaque where painted and transparent elsewhere; the top layer wins.

**segment.py** - put the fine detail (the pixels that change most when blurred) on the bottom layer, then reduce the rest to flat colours and deal them across the other layers, least busy on top. Outputs are numbered by step.

**text.py** - show the concept with a word made of blobs: each letter is a 5x5 grid pattern, and each filled cell becomes a blob on a randomly chosen layer. Each layer looks like random dots; the stack spells the word. Outputs are numbered by step.
