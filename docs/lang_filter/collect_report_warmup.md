# Collection report: `/work/nvme/bifq/project/LanguageConditionHJ/data/lang/warmup`

420 episodes in 30 shards, 0 skipped slots, status {'ok': 419, 'exception': 1}, disk 82.2 GB.

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
| T6 clean_pour | franka-panda | 30 | 0.03 |
| T6 clean_pour | ur5-wsg | 30 | 0.07 |
| T7 gap_place | franka-panda | 30 | 0.27 |
| T7 gap_place | ur5-wsg | 30 | 0.10 |
| T4 granular_carry | franka-panda | 30 | 0.80 |
| T4 granular_carry | ur5-wsg | 30 | 0.63 |
| T2 granular_pour | franka-panda | 30 | 0.47 |
| T2 granular_pour | ur5-wsg | 30 | 0.37 |
| T3 push_zone | franka-panda | 30 | 0.50 |
| T3 push_zone | ur5-wsg | 30 | 0.60 |
| T1 selective_avoid | franka-panda | 30 | 0.63 |
| T1 selective_avoid | ur5-wsg | 30 | 0.67 |
| T5 tray_transport | franka-panda | 30 | 0.97 |
| T5 tray_transport | ur5-wsg | 30 | 0.93 |

T1 hosts: place_bread_basket|franka-panda 6, place_bread_basket|ur5-wsg 6, place_burger_fries|franka-panda 6, place_burger_fries|ur5-wsg 6, place_container_plate|franka-panda 6, place_container_plate|ur5-wsg 6, stack_blocks_two|franka-panda 6, stack_blocks_two|ur5-wsg 6, stack_bowls_two|franka-panda 6, stack_bowls_two|ur5-wsg 6

| task | mean ticks | p95 | max | cap | at cap |
|---|---|---|---|---|---|
| selective_avoid | 244 | 358 | 380 | 750 | 0 |
| granular_pour | 448 | 507 | 597 | 750 | 0 |
| push_zone | 213 | 347 | 566 | 750 | 0 |
| granular_carry | 264 | 409 | 444 | 750 | 0 |
| tray_transport | 211 | 225 | 230 | 750 | 0 |
| clean_pour | 457 | 546 | 577 | 750 | 0 |
| gap_place | 377 | 411 | 431 | 750 | 0 |

## 2. Labels

Episodes with the rule violated, by designation (rate, n):

| task | consequence | harm | benign | unclear |
|---|---|---|---|---|
| selective_avoid | avoid | 0.83 (24) | 0.04 (122) | 0.85 (97) |
| selective_avoid | stay | 0.67 (24) | 0.03 (122) | 0.62 (97) |
| granular_pour | contain_particles | 0.90 (60) | – | – |
| push_zone | stay | 0.82 (22) | 0.00 (16) | – |
| push_zone | upright | 0.82 (22) | 0.00 (16) | – |
| push_zone | zone_exclude | 0.95 (82) | 0.00 (33) | – |
| push_zone | zone_include | – | 0.08 (60) | – |
| granular_carry | contain_particles | 0.70 (60) | – | – |
| tray_transport | upright | 0.72 (60) | 0.17 (90) | – |
| clean_pour | contain_particles | 0.97 (60) | – | – |
| clean_pour | exclude_particles | 0.83 (60) | 0.00 (21) | 0.08 (53) |
| gap_place | stay | 0.72 (60) | 0.01 (90) | – |
| gap_place | upright | 0.33 (60) | 0.01 (90) | – |

| task | harmful rule-rows | mixed rows | rows | sampleable rules / episode |
|---|---|---|---|---|
| selective_avoid | 14.0% | 57.8% | 14625 | 7.0 |
| granular_pour | 44.6% | 0.0% | 26868 | 1.0 |
| push_zone | 13.3% | 35.8% | 11027 | 3.3 |
| granular_carry | 55.2% | 0.0% | 15858 | 1.0 |
| tray_transport | 18.2% | 31.4% | 12638 | 2.5 |
| clean_pour | 25.4% | 47.5% | 27428 | 3.2 |
| gap_place | 5.9% | 19.8% | 22602 | 5.0 |

## 3. h quantiles (sampleable rules)

| consequence | at t=0 (1/5/25/50/75/95/99 %) | all ticks |
|---|---|---|
| avoid | +0.63 +1.00 +1.00 +1.00 +1.00 +1.00 +1.00 | -0.27 +0.02 +1.00 +1.00 +1.00 +1.00 +1.00 |
| contain_particles | +0.92 +0.93 +1.00 +1.00 +1.00 +1.00 +1.00 | -1.00 -1.00 -1.00 +0.94 +1.00 +1.00 +1.00 |
| exclude_particles | +1.00 +1.00 +1.00 +1.00 +1.00 +1.00 +1.00 | -0.84 -0.48 +0.07 +1.00 +1.00 +1.00 +1.00 |
| stay | +0.89 +0.96 +1.00 +1.00 +1.00 +1.00 +1.00 | -1.00 -1.00 +1.00 +1.00 +1.00 +1.00 +1.00 |
| upright | +1.00 +1.00 +1.00 +1.00 +1.00 +1.00 +1.00 | -1.00 -1.00 +1.00 +1.00 +1.00 +1.00 +1.00 |
| zone_exclude | +1.00 +1.00 +1.00 +1.00 +1.00 +1.00 +1.00 | -0.41 -0.14 +1.00 +1.00 +1.00 +1.00 +1.00 |
| zone_include | +0.19 +0.19 +0.20 +0.20 +0.20 +0.20 +0.20 | +0.17 +0.18 +0.20 +0.20 +0.20 +0.20 +0.20 |

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

## 5. Violation among non-ok episodes

- selective_avoid|benign: 0.00 of 6
- selective_avoid|unclear: 0.00 of 4

Examples (rule id: sentence):

**push_zone**

- `stay|obstacle_0|-`: don't slide the blue cylinder
- `upright|obstacle_0|-`: don't let the green tall cylinder fall over
- `upright|obstacle_0|-`: make sure the cup stays upright
- `outside|push_0|zone_red`: avoid pushing the coffee carton into the red zone
- `stay|obstacle_0|-`: leave the green cylinder where it is
- `inside|push_0|table_top`: make sure the box doesn't leave the tabletop
- `stay|obstacle_0|-`: leave the cylinder undisturbed
- `stay|obstacle_1|-`: leave the blue tall cylinder exactly where it is
- `outside|push_0|zone_yellow`: avoid pushing the green box into the yellow mat
- `outside|push_0|zone_yellow`: keep the blue box away from the yellow zone
- `stay|obstacle_0|-`: don't jostle the cup
- `inside|push_0|table_top`: keep the box on top of the tabletop
- `outside|push_0|zone_green`: don't run the coffee box over the green mat
- `inside|push_0|table_top`: make sure the coffee carton stays on the tabletop
- `upright|obstacle_0|-`: careful, don't knock the green tall cylinder down
- `stay|obstacle_0|-`: leave the cup where it is
- `outside|push_0|zone_black`: don't push the coffee box into the black zone
- `outside|push_0|zone_green`: don't slide the coffee box onto the green zone
- `stay|obstacle_0|-`: keep the cylinder undisturbed
- `upright|obstacle_0|-`: make sure the tall cylinder does not fall

**tray_transport**

- `upright|block_1|-`: don't let the short block topple
- `upright|block_0|-`: don't let the tall blue block go over
- `upright|block_0|-`: keep the tall yellow brick vertical
- `upright|block_2|-`: keep the red brick on its base
- `upright|block_2|-`: don't make the blue brick fall
- `upright|block_2|-`: don't let the blue brick topple
- `upright|block_1|-`: the short yellow block should stay upright
- `upright|block_2|-`: don't let the green brick go over
- `upright|block_1|-`: keep the red block on its base
- `upright|block_1|-`: don't topple the green brick
- `upright|block_0|-`: avoid knocking over the tall block
- `upright|block_0|-`: don't let the green brick fall over
- `upright|block_2|-`: don't make the yellow block fall
- `upright|block_1|-`: don't tip over the short green brick
- `upright|block_1|-`: don't topple the short yellow brick
- `upright|block_0|-`: don't tilt the tall blue brick over
- `upright|block_1|-`: don't let the short green brick lean over
- `upright|block_1|-`: don't push over the short brick
- `upright|block_1|-`: make sure the short blue brick doesn't topple
- `upright|block_1|-`: make sure the short block stays upright

**granular_carry**

- `particles_inside|beads|bowl_src_interior`: don't make a mess with the rice
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

**gap_place**

- `upright|block_0|-`: the blue block should stay upright
- `upright|block_1|-`: keep the yellow block on its base
- `upright|block_2|-`: don't topple the short brick
- `upright|block_0|-`: keep the tall brick from falling over
- `stay|block_1|-`: don't bump the yellow block
- `upright|block_0|-`: careful, don't knock the tall blue block down
- `stay|block_0|-`: leave the yellow brick exactly where it is
- `stay|block_0|-`: don't push the red brick
- `upright|block_1|-`: keep the yellow block from falling over
- `stay|block_1|-`: leave the green brick alone
- `upright|block_0|-`: keep the red brick standing up
- `upright|block_2|-`: don't tip over the short yellow block
- `upright|block_1|-`: don't knock over the tall green block
- `upright|block_2|-`: keep the green brick upright
- `stay|block_1|-`: keep the tall green brick where it is
- `stay|block_1|-`: don't shift the tall yellow block
- `upright|block_0|-`: keep the tall blue brick from falling over
- `upright|block_0|-`: be careful not to knock over the tall red brick
- `upright|block_0|-`: make sure the short yellow block stays upright
- `stay|block_1|-`: make sure the blue block stays put

**clean_pour**

- `particles_outside|beads|bin_yellow`: don't pour beans into the yellow tub
- `particles_inside|beads|bowl_src_interior+target_interior`: don't spill beans on the table
- `particles_outside|beads|bin_yellow`: make sure the yellow bin stays clean
- `particles_outside|beads|bin_yellow`: don't drop any beans into the tub
- `particles_outside|beads|plate_yellow`: no beads in the yellow plate
- `particles_inside|beads|bowl_src_interior+target_interior`: don't let any rice fall out
- `particles_inside|beads|bowl_src_interior+target_interior`: don't get beads on the table
- `particles_outside|beads|bin_yellow`: avoid getting rice in the yellow bin
- `particles_outside|beads|plate_green`: keep the beads out of the green plate
- `particles_inside|beads|bowl_src_interior+target_interior`: don't spill any rice
- `particles_inside|beads|bowl_src_interior+target_interior`: don't get beans on the table
- `particles_outside|beads|bin_green`: don't let rice into the green box
- `particles_outside|beads|bin_blue`: avoid getting rice in the blue box
- `particles_outside|beads|bin_yellow`: don't spill on the yellow bin
- `particles_outside|beads|plate_green`: don't spill on the green plate
- `particles_inside|beads|bowl_src_interior+target_interior`: keep every bit of the rice in
- `particles_outside|beads|plate_blue`: don't splash beads into the blue plate
- `particles_outside|beads|plate_blue`: don't let beans into the blue plate
- `particles_inside|beads|bowl_src_interior+target_interior`: don't get beans on the table
- `particles_outside|beads|bin_blue`: no beads in the box

**selective_avoid**

- `stay|obj_1|-`: don't nudge the drink box
- `avoid|obj_2|-`: don't brush against the drink box
- `avoid|obj_2|-`: move around the tin without touching it
- `avoid|obj_0|-`: stay clear of the tin
- `avoid|obj_0|-`: move around the carton without touching it
- `avoid|obj_3|-`: don't bump into the can
- `stay|obj_2|-`: leave the tin where it is
- `stay|obj_2|-`: keep the box where it is
- `stay|obj_2|-`: leave the tall cylinder alone
- `avoid|obj_2|-`: don't come into contact with the red box
- `avoid|obj_4|-`: do not touch the bottle
- `stay|obj_2|-`: don't knock the tall cylinder
- `stay|obj_0|-`: keep the box still
- `avoid|obj_0|-`: don't run into the can
- `avoid|obj_1|-`: avoid the can
- `avoid|obj_2|-`: keep a safe distance from the green tall cylinder
- `avoid|obj_1|-`: do not touch the drink box
- `avoid|obj_2|-`: don't brush against the black tall cylinder
- `stay|obj_2|-`: leave the water bottle undisturbed
- `avoid|obj_1|-`: be careful not to touch the yellow cylinder

**granular_pour**

- `particles_inside|beads|bowl_src_interior+target_interior`: make sure none of the beads falls out
- `particles_inside|beads|bowl_src_interior+target_interior`: don't spill beads on the table
- `particles_inside|beads|bowl_src_interior+target_interior`: make sure none of the rice ends up on the table
- `particles_inside|beads|bowl_src_interior+target_interior`: make sure none of the beans ends up on the table
- `particles_inside|beads|bowl_src_interior+target_interior`: make sure none of the beans ends up on the table
- `particles_inside|beads|bowl_src_interior+target_interior`: don't let any beans escape
- `particles_inside|beads|bowl_src_interior+target_interior`: don't make a mess with the beans
- `particles_inside|beads|bowl_src_interior+target_interior`: make sure none of the rice falls out
- `particles_inside|beads|bowl_src_interior+target_interior`: keep the beans in the container
- `particles_inside|beads|bowl_src_interior+target_interior`: don't spill the rice
- `particles_inside|beads|bowl_src_interior+target_interior`: don't spill beads on the table
- `particles_inside|beads|bowl_src_interior+target_interior`: don't spill beans on the table
- `particles_inside|beads|bowl_src_interior+target_interior`: keep the beans in the container
- `particles_inside|beads|bowl_src_interior+target_interior`: keep all the rice inside
- `particles_inside|beads|bowl_src_interior+target_interior`: don't spill the beans
- `particles_inside|beads|bowl_src_interior+target_interior`: keep every bit of the rice in
- `particles_inside|beads|bowl_src_interior+target_interior`: make sure none of the beads falls out
- `particles_inside|beads|bowl_src_interior+target_interior`: careful with the beads, don't spill it
- `particles_inside|beads|bowl_src_interior+target_interior`: keep the beads in the container
- `particles_inside|beads|bowl_src_interior+target_interior`: make sure none of the beans falls out

## 7. Wall time per tick (ms; build and wall in s per episode)

| task | render | encode | h | physics | capture | build s | wall s |
|---|---|---|---|---|---|---|---|
| selective_avoid | 326 | 241 | 27 | 5 | 620 | 12 | 147 |
| granular_pour | 271 | 232 | 26 | 11 | 560 | 12 | 261 |
| push_zone | 263 | 234 | 5 | 3 | 529 | 7 | 120 |
| granular_carry | 289 | 238 | 30 | 22 | 589 | 14 | 163 |
| tray_transport | 271 | 236 | 7 | 2 | 540 | 6 | 116 |
| clean_pour | 277 | 236 | 26 | 10 | 569 | 10 | 271 |
| gap_place | 249 | 225 | 6 | 2 | 505 | 7 | 194 |
