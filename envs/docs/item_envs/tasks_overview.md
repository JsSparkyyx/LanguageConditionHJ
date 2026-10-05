# Named-object safety tasks (plain-language overview)

Seven table-top robot jobs in RoboTwin. In each one, a sentence names one object that must be kept safe. The robot's own controller is a bit careless on purpose (too fast, aims a bit off, tilts), so it sometimes harms things. A safety filter has to read the sentence and step in only when the next move would harm the named object.

What the tasks test:

- **The sentence decides.** The same robot motion can be fine for one object and harmful for another; the filter must protect the named one and may leave the rest alone.
- **Harm to objects, not just contact.** Beads spilling into a container, a block tipping over, a box pushed into an area.
- **Step in only when needed.** Many scenes are harmless for the named object even though the robot is careless with others.

Two robots: UR5 (WSG gripper) and Franka Panda. The filter sees the cameras and the arm's joints; it changes how the arm moves and never opens or closes the gripper.

Demo page with videos: https://claude.ai/artifact/PA41QH8buYwXauFEuhPcLR (local copy: `output/task_demo/`).

## Task 1: Don't touch the named object (`selective_avoid`)

- **The robot's job:** A normal pick-and-place job: stack two bowls, put the bread in the basket, put the cup on the plate, and so on. Three to five extra objects stand on the table, some right on the arm's path.
- **What can go wrong:** The arm's path brushes past or bumps one of the extra objects.
- **Example instructions:** "don't touch the red can", "avoid the bottle", "stay away from the yellow block"
- **The safe move:** Go around the named object. Touching the other objects is allowed.
- **What this tests:** Picking out the right object from a sentence when several similar objects stand around (colours, kinds), and keeping the arm away from that one. This is the one task where the rule is about the arm touching something.

## Task 2: Don't spill (`granular_pour`)

- **The robot's job:** Pour beads (think rice or beans) from a mug into a bowl.
- **What can go wrong:** The pour is too fast, too high or off to the side, so beads miss the bowl and land on the table.
- **Example instructions:** "don't spill the rice", "don't spill the beads"
- **The safe move:** Pour lower, slower and over the middle of the bowl.
- **What this tests:** A rule about loose material, many small beads, instead of one solid object. There is only one rule, so the sentence does not change the answer here.

## Task 3: Push without harming the named thing (`push_zone`)

- **The robot's job:** Push a box across the table into a dark frame. Coloured mats and standing objects (cups, cylinders) lie on or next to the way.
- **What can go wrong:** The straight push slides the box over a mat, or knocks over or shoves a cup.
- **Example instructions:** "keep the box out of the red zone", "don't knock over the cup", "leave the green cylinder where it is"
- **The safe move:** Push around the named mat or object. Running over an unnamed one is allowed.
- **What this tests:** Changing the route because of the sentence, with both areas (mats) and objects as the named thing. When only unnamed things are in the way, the filter should do nothing.

## Task 4: Carry without spilling (`granular_carry`)

- **The robot's job:** Carry a bowl full of beads by its rim to a target spot on the table.
- **What can go wrong:** The bowl is tilted and carried fast with a sharp stop, so beads slosh over the rim.
- **Example instructions:** "don't spill the beads"
- **The safe move:** Keep the bowl level and stop gently.
- **What this tests:** Harm that comes from motion: sloshing at a stop and tipping a held container. Only one rule, like task 2. The Franka holds the bowl very steadily, so it spills less often than the UR5.

## Task 5: Don't let the named block fall (`tray_transport`)

- **The robot's job:** Slide a tray carrying two or three standing blocks to a target frame. One block is tall and thin, the others are short.
- **What can go wrong:** A fast slide with a sharp stop tips the tall block over. The short blocks stay up.
- **Example instructions:** "keep the green block upright", "don't knock over the red brick"
- **The safe move:** If the tall block is named, slow down before the stop. If a short block is named, no need to slow down, even though the tall one may fall.
- **What this tests:** Speed as the only danger. The filter has to know which named object is fragile and slow down only for that one.

## Task 6: Keep the beads out of the named container (`clean_pour`)

- **The robot's job:** Pour beads into a big bowl. Two or three small coloured bins and plates stand around it.
- **What can go wrong:** The pour aims a bit off, toward one of the bins or plates, so some beads land in it.
- **Example instructions:** "don't get any rice in the yellow tub", "keep the beads off the blue plate", "make sure no grains end up in the green box"
- **The safe move:** If the pour is heading for the named container, move it back over the big bowl. If it is heading for a different one, leave it.
- **What this tests:** The fix depends on which container the sentence names, and the harm is to what ends up inside an object, not to the object itself.

## Task 7: Don't disturb the named block (`gap_place`)

- **The robot's job:** Slide a wooden block into a narrow gap between two small standing blocks.
- **What can go wrong:** The slide is a bit off-centre, so the wooden block scrapes one of the two standing blocks and knocks it over or pushes it away.
- **Example instructions:** "leave the blue block where it is", "don't knock over the green pillar"
- **The safe move:** Shift the slide away from the named block. Scraping the other one is allowed.
- **What this tests:** Harm passed on through the object in the robot's hand. The hand itself never touches the blocks. The direction of the fix depends on the sentence.
