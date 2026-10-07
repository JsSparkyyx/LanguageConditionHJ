# Collection report: `/work/nvme/bifq/project/LanguageConditionHJ/data/lang/val`

70 episodes in 10 shards, 0 skipped slots, status {'ok': 70}, disk 13.5 GB.

**Flags**

- delay selective_avoid|avoid: median 0.64 s > 0.4 s
- delay selective_avoid|stay: median 0.68 s > 0.4 s
- delay granular_pour|contain_particles: median 1.52 s > 0.4 s
- delay push_zone|stay: median 3.04 s > 0.4 s
- delay push_zone|upright: median 3.40 s > 0.4 s
- delay push_zone|zone_exclude: median 1.68 s > 0.4 s
- delay push_zone|zone_include: median 1.94 s > 0.4 s
- delay granular_carry|contain_particles: median 0.72 s > 0.4 s
- delay tray_transport|upright: median 0.80 s > 0.4 s
- delay clean_pour|contain_particles: median 1.52 s > 0.4 s
- delay clean_pour|exclude_particles: median 1.60 s > 0.4 s
- delay gap_place|stay: median 0.64 s > 0.4 s
- delay gap_place|upright: median 0.60 s > 0.4 s

## 1. Episodes, success, length

| task | embodiment | episodes | success |
|---|---|---|---|
| T6 clean_pour | franka-panda | 5 | 0.00 |
| T6 clean_pour | ur5-wsg | 5 | 0.00 |
| T7 gap_place | franka-panda | 5 | 0.40 |
| T7 gap_place | ur5-wsg | 5 | 0.20 |
| T4 granular_carry | franka-panda | 5 | 0.80 |
| T4 granular_carry | ur5-wsg | 5 | 1.00 |
| T2 granular_pour | franka-panda | 5 | 0.60 |
| T2 granular_pour | ur5-wsg | 5 | 0.40 |
| T3 push_zone | franka-panda | 5 | 0.40 |
| T3 push_zone | ur5-wsg | 5 | 0.60 |
| T1 selective_avoid | franka-panda | 5 | 0.80 |
| T1 selective_avoid | ur5-wsg | 5 | 1.00 |
| T5 tray_transport | franka-panda | 5 | 0.80 |
| T5 tray_transport | ur5-wsg | 5 | 1.00 |

T1 hosts: place_bread_basket|franka-panda 1, place_bread_basket|ur5-wsg 1, place_burger_fries|franka-panda 1, place_burger_fries|ur5-wsg 1, place_container_plate|franka-panda 1, place_container_plate|ur5-wsg 1, stack_blocks_two|franka-panda 1, stack_blocks_two|ur5-wsg 1, stack_bowls_two|franka-panda 1, stack_bowls_two|ur5-wsg 1

| task | mean ticks | p95 | max | cap | at cap |
|---|---|---|---|---|---|
| selective_avoid | 242 | 315 | 325 | 750 | 0 |
| granular_pour | 436 | 485 | 495 | 750 | 0 |
| push_zone | 215 | 382 | 396 | 750 | 0 |
| granular_carry | 256 | 372 | 377 | 750 | 0 |
| tray_transport | 210 | 220 | 221 | 750 | 0 |
| clean_pour | 452 | 510 | 513 | 750 | 0 |
| gap_place | 371 | 418 | 424 | 750 | 0 |

## 2. Labels

Episodes with the rule violated, by designation (rate, n):

| task | consequence | harm | benign | unclear |
|---|---|---|---|---|
| selective_avoid | avoid | 1.00 (2) | 0.00 (22) | 0.94 (17) |
| selective_avoid | stay | 1.00 (2) | 0.00 (22) | 0.71 (17) |
| granular_pour | contain_particles | 1.00 (10) | – | – |
| push_zone | stay | 0.75 (4) | 0.00 (2) | – |
| push_zone | upright | 0.50 (4) | 0.00 (2) | – |
| push_zone | zone_exclude | 0.83 (12) | 0.00 (4) | – |
| push_zone | zone_include | – | 0.00 (10) | – |
| granular_carry | contain_particles | 0.60 (10) | – | – |
| tray_transport | upright | 0.70 (10) | 0.08 (13) | – |
| clean_pour | contain_particles | 1.00 (10) | – | – |
| clean_pour | exclude_particles | 0.80 (10) | 0.00 (8) | 0.00 (7) |
| gap_place | stay | 0.90 (10) | 0.00 (13) | – |
| gap_place | upright | 0.60 (10) | 0.00 (13) | – |

| task | harmful rule-rows | mixed rows | rows | sampleable rules / episode |
|---|---|---|---|---|
| selective_avoid | 12.4% | 58.4% | 2422 | 6.8 |
| granular_pour | 47.7% | 0.0% | 4361 | 1.0 |
| push_zone | 9.1% | 16.1% | 1940 | 2.6 |
| granular_carry | 43.3% | 0.0% | 2561 | 1.0 |
| tray_transport | 16.5% | 33.4% | 2100 | 2.3 |
| clean_pour | 22.5% | 49.7% | 4523 | 3.5 |
| gap_place | 9.1% | 24.8% | 3710 | 4.6 |

## 3. h quantiles (sampleable rules)

| consequence | at t=0 (1/5/25/50/75/95/99 %) | all ticks |
|---|---|---|
| avoid | +0.73 +1.00 +1.00 +1.00 +1.00 +1.00 +1.00 | -0.20 +0.12 +1.00 +1.00 +1.00 +1.00 +1.00 |
| contain_particles | +0.93 +0.93 +1.00 +1.00 +1.00 +1.00 +1.00 | -1.00 -1.00 -1.00 +1.00 +1.00 +1.00 +1.00 |
| exclude_particles | +1.00 +1.00 +1.00 +1.00 +1.00 +1.00 +1.00 | -0.74 -0.18 +0.08 +1.00 +1.00 +1.00 +1.00 |
| stay | +0.92 +0.96 +1.00 +1.00 +1.00 +1.00 +1.00 | -1.00 -1.00 +0.99 +1.00 +1.00 +1.00 +1.00 |
| upright | +0.99 +1.00 +1.00 +1.00 +1.00 +1.00 +1.00 | -1.00 -1.00 +1.00 +1.00 +1.00 +1.00 +1.00 |
| zone_exclude | +1.00 +1.00 +1.00 +1.00 +1.00 +1.00 +1.00 | -0.26 -0.06 +1.00 +1.00 +1.00 +1.00 +1.00 |
| zone_include | +0.19 +0.19 +0.19 +0.20 +0.20 +0.20 +0.20 | +0.18 +0.18 +0.19 +0.19 +0.20 +0.20 +0.20 |

## 4. Delay audit (risky-segment start -> first violation)

| task / consequence | n | median s | p25 | p75 | no risky segment before | flag (> 0.4 s) |
|---|---|---|---|---|---|---|
| selective_avoid|avoid | 40 | 0.64 | 0.48 | 1.09 | 16 | **yes** |
| selective_avoid|stay | 43 | 0.68 | 0.52 | 1.10 | 5 | **yes** |
| granular_pour|contain_particles | 19 | 1.52 | 0.54 | 1.86 | 0 | **yes** |
| push_zone|stay | 9 | 3.04 | 1.96 | 3.36 | 0 | **yes** |
| push_zone|upright | 9 | 3.40 | 2.04 | 5.76 | 0 | **yes** |
| push_zone|zone_exclude | 28 | 1.68 | 1.01 | 2.33 | 0 | **yes** |
| push_zone|zone_include | 2 | 1.94 | 1.37 | 2.51 | 0 | **yes** |
| granular_carry|contain_particles | 7 | 0.72 | 0.54 | 0.84 | 1 | **yes** |
| tray_transport|upright | 13 | 0.80 | 0.36 | 0.92 | 0 | **yes** |
| clean_pour|contain_particles | 19 | 1.52 | 1.28 | 1.82 | 0 | **yes** |
| clean_pour|exclude_particles | 13 | 1.60 | 1.56 | 2.28 | 0 | **yes** |
| gap_place|stay | 19 | 0.64 | 0.58 | 0.76 | 0 | **yes** |
| gap_place|upright | 10 | 0.60 | 0.53 | 0.68 | 0 | **yes** |

Examples (rule id: sentence):

**push_zone**

- `outside|push_0|zone_green`: keep the coffee box outside the green zone
- `inside|push_0|table_top`: the coffee carton should stay on the table
- `outside|push_0|zone_green`: don't move the coffee carton into the green mat
- `upright|obstacle_0|-`: don't let the cup topple
- `stay|obstacle_0|-`: keep the cylinder where it is
- `stay|obstacle_0|-`: don't knock the tall cylinder out of place
- `inside|push_0|table_top`: don't drop the green box off the tabletop
- `outside|push_0|zone_yellow`: keep the box clear of the yellow mat
- `upright|obstacle_0|-`: don't tip over the cup
- `upright|obstacle_0|-`: careful, don't knock the cup down
- `stay|obstacle_0|-`: don't shove the black tall cylinder
- `upright|obstacle_0|-`: make sure the cylinder doesn't topple
- `inside|push_0|table_top`: make sure the green box stays on the tabletop
- `inside|push_0|table_top`: keep the green box away from the edge of the table
- `stay|obstacle_0|-`: make sure the cup doesn't move
- `outside|push_0|zone_green`: don't push the coffee box into the green mat
- `outside|push_0|zone_black`: don't slide the coffee box onto the black zone
- `stay|obstacle_0|-`: keep the cylinder undisturbed
- `stay|obstacle_0|-`: avoid moving the tall cylinder
- `stay|obstacle_0|-`: leave the cup alone

**tray_transport**

- `upright|block_1|-`: don't let the short brick tip over
- `upright|block_1|-`: avoid knocking over the short yellow brick
- `upright|block_2|-`: keep the blue brick on its base
- `upright|block_2|-`: don't make the red brick fall
- `upright|block_1|-`: don't let the short brick topple
- `upright|block_1|-`: the short green block should stay upright
- `upright|block_1|-`: don't let the blue brick go over
- `upright|block_1|-`: keep the yellow block on its base
- `upright|block_1|-`: don't topple the green brick
- `upright|block_0|-`: avoid knocking over the tall block
- `upright|block_1|-`: don't let the green brick fall over
- `upright|block_2|-`: don't make the green block fall
- `upright|block_1|-`: don't tip over the short green brick
- `upright|block_0|-`: don't topple the tall red brick
- `upright|block_0|-`: don't tilt the tall red brick over
- `upright|block_1|-`: don't let the short green brick lean over
- `upright|block_1|-`: don't push over the short brick
- `upright|block_1|-`: make sure the short green brick doesn't topple
- `upright|block_1|-`: make sure the short block stays upright
- `upright|block_0|-`: don't make the tall brick fall

**granular_carry**

- `particles_inside|beads|bowl_src_interior`: avoid spilling the beans
- `particles_inside|beads|bowl_src_interior`: don't get rice on the table
- `particles_inside|beads|bowl_src_interior`: don't lose any beans
- `particles_inside|beads|bowl_src_interior`: careful with the rice, don't spill it
- `particles_inside|beads|bowl_src_interior`: don't waste any beads
- `particles_inside|beads|bowl_src_interior`: keep all the rice inside
- `particles_inside|beads|bowl_src_interior`: don't waste any beads
- `particles_inside|beads|bowl_src_interior`: don't let any beans escape
- `particles_inside|beads|bowl_src_interior`: don't spill beans on the table
- `particles_inside|beads|bowl_src_interior`: don't spill any rice
- `particles_inside|beads|bowl_src_interior`: no rice on the table
- `particles_inside|beads|bowl_src_interior`: don't let any rice fall out
- `particles_inside|beads|bowl_src_interior`: make sure none of the beads ends up on the table
- `particles_inside|beads|bowl_src_interior`: don't let the beans spill
- `particles_inside|beads|bowl_src_interior`: avoid spilling the beans
- `particles_inside|beads|bowl_src_interior`: keep all the beads inside
- `particles_inside|beads|bowl_src_interior`: don't scatter the beans
- `particles_inside|beads|bowl_src_interior`: don't spill any rice
- `particles_inside|beads|bowl_src_interior`: don't drop any beans
- `particles_inside|beads|bowl_src_interior`: don't let the rice spill

**gap_place**

- `upright|block_0|-`: the tall blue brick should stay upright
- `upright|block_0|-`: don't let the blue brick tip over
- `upright|block_0|-`: don't make the tall blue brick fall
- `stay|block_1|-`: don't bump the green block
- `upright|block_0|-`: careful, don't knock the tall blue block down
- `stay|block_1|-`: leave the blue brick exactly where it is
- `stay|block_0|-`: don't push the red brick
- `upright|block_1|-`: keep the yellow block from falling over
- `stay|block_1|-`: leave the yellow brick alone
- `upright|block_0|-`: keep the blue brick standing up
- `upright|block_1|-`: don't tip over the tall yellow block
- `upright|block_1|-`: don't knock over the tall block
- `upright|block_1|-`: keep the red brick upright
- `stay|block_1|-`: keep the tall brick where it is
- `stay|block_2|-`: don't shift the tall yellow block
- `upright|block_0|-`: keep the tall brick from falling over
- `upright|block_1|-`: be careful not to knock over the tall red brick
- `upright|block_0|-`: make sure the tall blue block stays upright
- `stay|block_1|-`: make sure the blue block stays put
- `upright|block_2|-`: keep the tall blue brick from falling over

**clean_pour**

- `particles_inside|beads|bowl_src_interior+target_interior`: don't let any beans escape
- `particles_outside|beads|bin_green`: make sure the green bin stays clean
- `particles_outside|beads|bin_blue`: don't drop any beans into the blue tub
- `particles_outside|beads|plate_yellow`: no beads in the yellow plate
- `particles_inside|beads|bowl_src_interior+target_interior`: don't let any rice fall out
- `particles_inside|beads|bowl_src_interior+target_interior`: don't get beads on the table
- `particles_outside|beads|plate_blue`: avoid getting rice in the plate
- `particles_outside|beads|bin_green`: keep the beads out of the green box
- `particles_inside|beads|bowl_src_interior+target_interior`: don't spill any rice
- `particles_inside|beads|bowl_src_interior+target_interior`: don't get beans on the table
- `particles_outside|beads|bin_yellow`: don't let rice into the yellow box
- `particles_outside|beads|bin_yellow`: avoid getting rice in the box
- `particles_outside|beads|plate_blue`: don't spill on the blue plate
- `particles_outside|beads|bin_blue`: don't spill on the blue bin
- `particles_inside|beads|bowl_src_interior+target_interior`: keep every bit of the rice in
- `particles_outside|beads|plate_blue`: don't splash beads into the blue plate
- `particles_outside|beads|plate_blue`: don't let beans into the blue plate
- `particles_inside|beads|bowl_src_interior+target_interior`: don't get beans on the table
- `particles_outside|beads|plate_blue`: no beads in the blue plate
- `particles_outside|beads|plate_yellow`: don't spill any beads into the yellow dish

**selective_avoid**

- `avoid|obj_1|-`: don't touch the can
- `avoid|obj_4|-`: never touch the tin
- `stay|obj_2|-`: don't jostle the tin
- `avoid|obj_3|-`: leave the box untouched
- `avoid|obj_1|-`: go around the can
- `stay|obj_0|-`: leave the black cylinder where it is
- `avoid|obj_4|-`: keep the arm away from the tin
- `stay|obj_2|-`: keep the drink box where it is
- `avoid|obj_2|-`: don't knock into the drink box
- `avoid|obj_2|-`: keep away from the can
- `stay|obj_0|-`: make sure the coffee box doesn't move
- `stay|obj_0|-`: don't displace the cube
- `stay|obj_1|-`: don't displace the drink box
- `stay|obj_2|-`: don't shift the block
- `avoid|obj_2|-`: never touch the bottle
- `avoid|obj_3|-`: don't touch the box
- `avoid|obj_4|-`: don't make contact with the can
- `avoid|obj_2|-`: don't knock into the can
- `avoid|obj_0|-`: stay clear of the tall cylinder
- `avoid|obj_2|-`: keep away from the can

**granular_pour**

- `particles_inside|beads|bowl_src_interior+target_interior`: keep the rice in the container
- `particles_inside|beads|bowl_src_interior+target_interior`: careful with the beads, don't spill it
- `particles_inside|beads|bowl_src_interior+target_interior`: keep all the beads inside
- `particles_inside|beads|bowl_src_interior+target_interior`: keep the rice in the container
- `particles_inside|beads|bowl_src_interior+target_interior`: don't let any beads fall out
- `particles_inside|beads|bowl_src_interior+target_interior`: make sure the rice stays in
- `particles_inside|beads|bowl_src_interior+target_interior`: don't get beads on the table
- `particles_inside|beads|bowl_src_interior+target_interior`: make sure none of the beads ends up on the table
- `particles_inside|beads|bowl_src_interior+target_interior`: be careful not to spill the rice
- `particles_inside|beads|bowl_src_interior+target_interior`: make sure none of the rice falls out
- `particles_inside|beads|bowl_src_interior+target_interior`: careful with the beans, don't spill it
- `particles_inside|beads|bowl_src_interior+target_interior`: keep the beans in the container
- `particles_inside|beads|bowl_src_interior+target_interior`: keep all the rice inside
- `particles_inside|beads|bowl_src_interior+target_interior`: don't spill the beans
- `particles_inside|beads|bowl_src_interior+target_interior`: keep every bit of the rice in
- `particles_inside|beads|bowl_src_interior+target_interior`: make sure none of the beads falls out
- `particles_inside|beads|bowl_src_interior+target_interior`: careful with the beads, don't spill it
- `particles_inside|beads|bowl_src_interior+target_interior`: keep the beads in the container
- `particles_inside|beads|bowl_src_interior+target_interior`: make sure none of the beans falls out
- `particles_inside|beads|bowl_src_interior+target_interior`: no rice on the table

## 7. Wall time per tick (ms; build and wall in s per episode)

| task | render | encode | h | physics | capture | build s | wall s |
|---|---|---|---|---|---|---|---|
| selective_avoid | 325 | 226 | 326 | 4 | 899 | 14 | 224 |
| granular_pour | 316 | 233 | 328 | 10 | 907 | 23 | 407 |
| push_zone | 309 | 237 | 277 | 2 | 847 | 16 | 195 |
| granular_carry | 333 | 231 | 341 | 19 | 934 | 21 | 242 |
| tray_transport | 318 | 233 | 291 | 2 | 865 | 11 | 184 |
| clean_pour | 321 | 236 | 338 | 9 | 923 | 27 | 427 |
| gap_place | 301 | 222 | 279 | 2 | 825 | 14 | 313 |
