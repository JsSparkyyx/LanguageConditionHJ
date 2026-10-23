"""My notes on the constraint templates (envs/main/item_envs/language/paraphrases.yaml, 2026-10-20), for the review
page. Level: "problem" = the sentence says something other than what the rule measures; "check" = vague, odd, or
a grammar slip. Unlisted templates read fine to me."""

P, C = "problem", "check"

RULES = {
    "avoid": {"tasks": "T1 selective_avoid",
              "measures": "Neither the arm nor the object it is holding may come within 1 cm of the named object's real "
                          "shape. Touching it is the violation; moving it is not required.",
              "scale": "distance, 5 cm = comfortably safe"},
    "stay": {"tasks": "T1 selective_avoid, T3 push_zone, T7 gap_place",
             "measures": "The named object must not be moved more than 3 cm (T1), 1.5 cm (T3) or 2 cm (T7); tilting "
                         "it counts as moving it (a 45 degree tilt counts as the whole limit in T3 and T7, 20 degrees in "
                         "T1). Touching it, or a small push that moves it less than the limit, is allowed. In T1 the "
                         "same object always also has an avoid rule (don't touch).",
             "scale": "displacement against the limit"},
    "upright": {"tasks": "T3 push_zone, T5 tray_transport, T7 gap_place",
                "measures": "The named object must not tilt more than 45 degrees (2026-10-23; 10 before). In T3 and T7 "
                            "the same object also has a stay rule (don't move it).",
                "scale": "tilt against 45 degrees"},
    "zone_exclude": {"tasks": "T3 push_zone",
                     "measures": "The box being pushed must never enter the named mat, at any moment (passing over it "
                                 "counts, not only where it ends).",
                     "scale": "distance, 5 cm"},
    "zone_include": {"tasks": "T3 push_zone",
                     "measures": "The box being pushed must stay on the table (the only region of this rule).",
                     "scale": "distance to the edge, 5 cm"},
    "contain_particles": {"tasks": "T6 clean_pour (the spill rule); T2 granular_pour and T4 granular_carry, which are "
                                   "not used from 2026-10-23 / 2026-10-22",
                          "measures": "Never named from 2026-10-23 (Yuxuan): this rule has no sentences, but its h is "
                                      "still recorded. No bead may land outside the containers. T4: the beads stay in "
                                      "the carried bowl. T2 and T6: the beads are POURED from the mug into the target "
                                      "bowl on purpose; a bead in the mug or the target bowl is fine, a bead anywhere "
                                      "else is the violation (on the table, or in T6 in one of the other two bowls).",
                          "scale": "bead count once violated (1 bead = -0.5, 20+ = -1)"},
    "exclude_particles": {"tasks": "T6 clean_pour",
                          "measures": "No bead may land in the named bowl: one of the two bowls that are not the pour "
                                      "target (the beads are meant for the target bowl).",
                          "scale": "bead count once violated"},
}

FLAGS = {
    ("avoid", "keep the arm away from {obj}"): (C, "The rule also covers the object in the hand; this names the arm only."),
    ("avoid", "keep a safe distance from {obj}"): (C, "Suggests a large gap; the rule only needs 1 cm."),
    ("avoid", "watch out for {obj}"): (P, "Says nothing about what must not happen."),
    ("avoid", "give {obj} a wide berth"): (C, "Suggests a large gap; the rule only needs 1 cm."),
    ("avoid", "leave {obj} untouched"): (C, "Close to the stay templates 'leave {obj} alone / undisturbed' for the same object."),
    ("stay", "don't disturb {obj}"): (P, "Reads as 'don't touch it'. That is the avoid rule on the same object (T1)."),
    ("stay", "don't push {obj}"): (P, "Forbids any push; the rule allows a push that moves it less than the limit."),
    ("stay", "leave {obj} alone"): (P, "Reads as 'don't touch it' (avoid rule on the same object in T1)."),
    ("stay", "don't nudge {obj}"): (P, "A nudge is a small movement, which the rule allows (limit 1.5-3 cm)."),
    ("stay", "don't bump {obj}"): (P, "Bumping is touching; the avoid rule on the same object means exactly that."),
    ("stay", "leave {obj} undisturbed"): (P, "Reads as 'don't touch it'."),
    ("stay", "don't jostle {obj}"): (P, "Jostling is bumping into it; the rule is about moving it."),
    ("stay", "don't knock {obj}"): (P, "Knocking is touching; also close to the upright template 'don't knock {obj} down'."),
    ("stay", "keep {obj} undisturbed"): (P, "Reads as 'don't touch it'."),
    ("stay", "let {obj} be"): (P, "Reads as 'don't touch it'."),
    ("stay", "leave {obj} exactly where it is"): (C, "'Exactly' is stricter than the 1.5-3 cm limit."),
    ("stay", "don't make {obj} budge"): (C, "'Budge' means even a tiny move; the rule allows up to 1.5-3 cm."),
    ("stay", "don't rearrange {obj}"): (C, "Odd for a single object."),
    ("upright", "don't make {obj} fall"): (C, "'Fall' can also mean falling off the tray (T5) or out of the hand."),
    ("upright", "make sure {obj} does not fall"): (C, "Same: 'fall' is wider than tipping over."),
    ("upright", "{obj} must not fall"): (C, "Same: 'fall' is wider than tipping over."),
    ("upright", "don't let {obj} go over"): (P, "Unclear; 'go over' can mean several things."),
    ("upright", "careful, don't knock {obj} down"): (C, "Close to the stay template 'don't knock {obj}'."),
    ("upright", "don't upset {obj}"): (C, "Old-fashioned for 'tip over'; most readers take it as 'disturb'."),
    ("zone_exclude", "don't put {obj} in {region}"): (C, "'Put' is placing; in T3 the box is pushed."),
    ("zone_exclude", "don't run {obj} over {region}"): (C, "'Run over' is what a vehicle does; odd here."),
    ("zone_exclude", "make sure {obj} doesn't end up in {region}"): (P, "'End up' suggests only the final position counts; "
                                                                          "passing over the mat already breaks the rule."),
    ("zone_exclude", "{obj} has no business in {region}"): (C, "Idiom; fine for a held-out template if you like it."),
    ("zone_include", "keep {obj} inside {region}"): (C, "'Inside the table' is odd; the region is always the table."),
    ("contain_particles", "keep the {material} in the container"): (P, "Wrong for T2 / T6, where the beads are poured out "
                                                                       "of the mug on purpose; also 'the container' is "
                                                                       "ambiguous with two containers."),
    ("contain_particles", "don't let any {material} fall out"): (P, "In T2 / T6 beads fall out of the mug by design."),
    ("contain_particles", "don't waste any {material}"): (C, "Vague; could mean 'pour all of them'."),
    ("contain_particles", "keep all the {material} inside"): (P, "Wrong for the pouring tasks (inside what?)."),
    ("contain_particles", "make sure none of the {material} falls out"): (P, "In T2 / T6 beads fall out of the mug by design."),
    ("contain_particles", "make sure the {material} stays in"): (P, "Grammar ('the beads stays in') and wrong for the pouring tasks."),
    ("contain_particles", "don't lose any {material}"): (C, "Vague."),
    ("contain_particles", "careful with the {material}, don't spill it"): (C, "Grammar: 'it' for a plural word ('the beads')."),
    ("contain_particles", "keep every bit of the {material} in"): (P, "Wrong for the pouring tasks (in what?)."),
    ("contain_particles", "the {material} must stay in the container"): (P, "Same as 'keep the beads in the container'."),
    ("contain_particles", "don't let the {material} overflow"): (C, "Overflow is a different event (filling past the rim)."),
    ("contain_particles", "keep the {material} contained"): (C, "Vague for the pouring tasks."),
    ("contain_particles", "don't splash the {material}"): (C, "Splashing is for liquids."),
    ("exclude_particles", "keep {region} clean"): (P, "Does not say clean of what (the beads)."),
    ("exclude_particles", "keep {region} empty"): (P, "Does not mention the beads; the bin may hold other things."),
    ("exclude_particles", "make sure {region} stays clean"): (P, "Does not say clean of what."),
    ("exclude_particles", "{region} must stay clean"): (P, "Does not say clean of what."),
    ("exclude_particles", "don't spill on {region}"): (P, "No material named, and 'on' a bin is odd."),
    ("exclude_particles", "make sure none of the {material} ends up in {region}"): (C, "'Ends up' suggests only the end "
                                                                                     "counts; a bead entering counts at any moment."),
    ("exclude_particles", "make sure no {material} end up in {region}"): (C, "Same 'end up' remark."),
    ("exclude_particles", "don't splash {material} into {region}"): (C, "Splashing is for liquids."),
    ("exclude_particles", "not a single bead in {region}"): (C, "Says 'bead' whatever the material word is."),
    ("exclude_particles", "spare {region} from the {material}"): (C, "Unusual phrasing."),
}

CROSS = [
    ("problem", "Stay sentences that mean 'don't touch'",
     "{n_stay} of the 30 stay templates (bump, nudge, knock, jostle, disturb, leave alone, ...) say 'don't touch it'. In T1 the "
     "same object also has an avoid rule, which is exactly 'don't touch it', while stay allows touching as long as the "
     "object moves less than 3 cm. A sentence then does not tell the two rules apart, and the model is trained to give "
     "them different values."),
    ("problem", "Pouring tasks told to keep the beads in",
     "In T2 and T6 the beads are poured out of the mug into the bowl on purpose. {n_pour} spill templates say 'keep the beads "
     "in the container / inside / stays in / don't let them fall out', which reads as 'don't pour'. In T4 (carry) they "
     "are right."),
    ("problem", "Bins and plates with no bead named",
     "'Keep the blue bin clean / empty' ({n_clean} templates) never says clean of what."),
    ("check", "'End up' while the rule checks every moment",
     "Zone and bin templates with 'end up' suggest only where the box or beads finish; the rules count passing through."),
    ("check", "Plural grammar",
     "The bead words are plural (beads, balls, small balls): 'make sure the beads stays in', 'don't spill it'."),
    ("check", "Words for regions",
     "Bins are also called 'box', the same word as the boxes in T1 / T3. Old recordings (before 10-12) keep their "
     "stored region words: the example sentences below still say 'saucer'; a retrain on those recordings would need the "
     "region words renamed too, as the object words are."),
]
