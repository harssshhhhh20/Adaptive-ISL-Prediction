# Error analysis examples (seed 0, LOSO)

## L_trel_gate|clean — WER 84.9

**correct**

- `serve the food|7`  REF: SERVE FOOD  |  HYP: SERVE FOOD
- `i need water|1`  REF: I NEED WATER  |  HYP: I NEED WATER
- `prepare the bed|4`  REF: PREPARE BED  |  HYP: PREPARE BED
- `prepare the bed|2`  REF: PREPARE BED  |  HYP: PREPARE BED
- `i am crying|3`  REF: I CRY  |  HYP: I CRY
- `wear the shirt|7`  REF: WEAR SHIRT  |  HYP: WEAR SHIRT

**one_error**

- `who are you|4`  REF: WHO YOU  |  HYP: WHAT YOU
- `i am hungry|5`  REF: I HUNGRY  |  HYP: I NEED
- `i promise|4`  REF: I PROMISE  |  HYP: I BE
- `i am tired|2`  REF: I TIRED  |  HYP: I HELP
- `i am tired|7`  REF: I TIRED  |  HYP: I PROMISE
- `go and sleep|7`  REF: GO SLEEP  |  HYP: NOT SLEEP

**bad**

- `i am feeling bored|3`  REF: I FEELING BORED  |  HYP: YOU HIDE
- `what happened|4`  REF: WHAT HAPPENED  |  HYP: WHY YOU
- `speak softly|5`  REF: SPEAK SOFTLY  |  HYP: WHAT YOU
- `you do anything, i do not care|2`  REF: YOU DO ANYTHING I NOT CARE  |  HYP: WE BED
- `shall we go outside|5`  REF: WE GO OUTSIDE  |  HYP: YOU WELCOME
- `do not worry|3`  REF: NOT WORRY  |  HYP: I NOT FAVOUR

## G_concat|clean — WER 84.6

**correct**

- `you are welcome|2`  REF: YOU WELCOME  |  HYP: YOU WELCOME
- `prepare the bed|6`  REF: PREPARE BED  |  HYP: PREPARE BED
- `you are bad|3`  REF: YOU BAD  |  HYP: YOU BAD
- `wear the shirt|6`  REF: WEAR SHIRT  |  HYP: WEAR SHIRT
- `comb your hair|1`  REF: COMB YOU HAIR  |  HYP: COMB YOU HAIR
- `i am crying|3`  REF: I CRY  |  HYP: I CRY

**one_error**

- `i got hurt|2`  REF: I GOT HURT  |  HYP: I HURT HURT
- `how are things|7`  REF: HOW THINGS  |  HYP: YOU THINGS
- `i am hungry|7`  REF: I HUNGRY  |  HYP: I AFRAID
- `i am (age)|3`  REF: I AGE  |  HYP: I CRY
- `i am crying|7`  REF: I CRY  |  HYP: I FEELING
- `you are welcome|5`  REF: YOU WELCOME  |  HYP: YOU PLEASE

**bad**

- `nice to meet you|5`  REF: NICE MEET YOU  |  HYP: I NEED
- `what did you tell him|7`  REF: WHAT YOU TELL HIM  |  HYP: SPEAK SOFTLY
- `what do you think|2`  REF: WHAT DO YOU THINK  |  HYP: YOUR TO
- `he she is my friend|2`  REF: HE SHE MY FRIEND  |  HYP: I VERY
- `are you free today|5`  REF: YOU FREE TODAY  |  HYP: I NOT
- `how are things|5`  REF: HOW THINGS  |  HYP: HOW NOT NEED

## A_rgb|clean — WER 84.0

**correct**

- `i am suffering from fever|4`  REF: I SUFFERING FROM FEVER  |  HYP: I SUFFERING FROM FEVER
- `i am suffering from fever|6`  REF: I SUFFERING FROM FEVER  |  HYP: I SUFFERING FROM FEVER
- `prepare the bed|6`  REF: PREPARE BED  |  HYP: PREPARE BED
- `take care of yourself|3`  REF: TAKE CARE OF YOURSELF  |  HYP: TAKE CARE OF YOURSELF
- `congratulations|3`  REF: CONGRATULATIONS  |  HYP: CONGRATULATIONS
- `let him take time|4`  REF: LET HIM TAKE TIME  |  HYP: LET HIM TAKE TIME

**one_error**

- `what you want|2`  REF: WHAT YOU WANT  |  HYP: YOU WANT
- `i am (age)|4`  REF: I AGE  |  HYP: I
- `what did you tell him|3`  REF: WHAT YOU TELL HIM  |  HYP: YOU TELL HIM
- `you are bad|2`  REF: YOU BAD  |  HYP: YOU
- `congratulations|4`  REF: CONGRATULATIONS  |  HYP: GO
- `i am hungry|2`  REF: I HUNGRY  |  HYP: HUNGRY HUNGRY

**bad**

- `how are things|3`  REF: HOW THINGS  |  HYP: ∅
- `i enjoyed a lot|4`  REF: I ENJOYED A LOT  |  HYP: I VERY HAPPY THE HAPPY
- `how dare you|6`  REF: HOW DARE YOU  |  HYP: NOT IT
- `he is going into the room|1`  REF: HE GO INTO ROOM  |  HYP: ∅
- `i am really grateful|3`  REF: I REALLY GRATEFUL  |  HYP: YOU HEART
- `are you hiding something|5`  REF: YOU HIDE SOMETHING  |  HYP: ME
