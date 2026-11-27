# Writing the scene prompt

The model takes the mouth from the audio and everything else from the photo plus this prompt. A vague prompt gives a stiff talking head. A concrete one gives eyebrows, nods, hand gestures and a background that breathes.

## Recipe (one paragraph, 50–90 words, English)

1. **Who:** age range, role, clothing, what they hold.
2. **Where:** setting and light, taken from the photo; don't invent a new place.
3. **How they talk:** "speaking directly to the camera", plus 2–4 concrete behaviours: raises eyebrows, nods slightly, tilts head, gestures with the free hand, moves hands on the desk, smiles briefly.
4. **Secondary motion:** hair moving in a breeze, rain on the street, a car passing, people far in the background.
5. **Camera:** "steady medium shot" (or close-up). Ask for a still camera unless the user wants motion.

Avoid lip or phoneme descriptions, emotional extremes ("screaming", "laughing hard") unless the audio has them, and text you want legible on screen. The model can't render readable new text, though existing logos in the photo usually survive.

## What the models actually read

- **InfiniteTalk (fast LoRA, cfg 1):** a plain negative prompt is never evaluated at cfg 1. Use `--negative "..."`: talkframe applies it with NAG (Normalized Attention Guidance) inside attention, at no extra sampling cost. Keep it short and concrete (`"wristwatch, bracelet, dental braces"`); `--nag-scale` (default 11) sets its strength.
- **LongCat (distilled):** text guidance is forced to 1 and there is no NAG, so it has no negative prompt at all. Its README asks for long, detailed prompts (appearance, clothing, actions, setting) with explicit "speaking/talking" verbs.
- **Both:** say what you want instead of what you don't. Naming an object ("no watch") can make the model draw it; "bare forearms and wrists" doesn't.

## Taming common artifacts

| Problem | Prompt (positive) | `--negative` (InfiniteTalk) | Photo |
|---|---|---|---|
| Invented watch or bracelet when a hand lifts | "bare forearms and wrists" | `wristwatch, bracelet` | Both wrists visible and bare. A hand covering the other wrist makes the model guess what's under it |
| Exaggerated mouth, big lower teeth | "speaks calmly with subtle, relaxed lip movements, mostly her upper teeth visible" | `wide open mouth, exaggerated mouth movements` | A closed or softly closed mouth in the photo |
| Teeth that look like braces or mosaic | "natural, clean white teeth" | `dental braces` | Teeth sharp and evenly lit, or mouth closed |
| Hands melting or changing | "small, slow gestures, hands mostly resting on the desk" | `extra fingers, fused fingers` | Hands fully in frame, fingers separated |
| Hair or background swimming | "her hair stays mostly still", "static camera" | — | — |

Tested 2026-11-27 (desk presenter, one hand covering the other wrist): with resting hands there was no watch, but asking for gestures brought a smartwatch back even with `--negative "wristwatch, bracelet"` at `--nag-scale 11`. A negative prompt helps at the margins; the photo decides. If a detail must never appear, show the area it would cover in the photo.

Don't ban "jewelry" in general when the photo has earrings or a nose ring: the model will fight the photo. Name the specific item you don't want.

## Examples

**Street reporter (photo: woman on a rainy street holding a branded mic):**
> A female TV news reporter in her thirties stands on a rainy city street, speaking directly to the camera while holding a microphone with a station logo. She talks with natural, expressive facial expressions, raises her eyebrows, nods slightly, and gestures with her free hand to explain. Her long wavy hair moves gently in the breeze. Natural daylight, realistic, steady medium shot.

**Desk presenter (photo: woman at a desk, bookshelves behind):**
> A female news presenter sits at a wooden desk in a cozy studio with bookshelves, speaking directly to the camera into a desk microphone. She talks with natural, expressive facial expressions, raises her eyebrows, tilts her head, and moves her hands on the desk while explaining. Her long wavy hair moves slightly. Warm lighting, realistic, steady medium shot.

**Teacher, close-up (photo: man in a classroom):**
> A middle-aged male teacher in a classroom speaks warmly to the camera, smiling occasionally, nodding as he makes a point and lifting one hand to emphasise key ideas. A whiteboard and students' desks are softly blurred behind him. Soft daylight from a window, realistic, steady close-up shot.
