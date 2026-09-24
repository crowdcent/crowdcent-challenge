# 3. The submission: your predictions "today's quiz"

This is everything you hand in.

> These numbers are randomly generated: they show the *shape* of a submission, not any real model's opinion. Three columns, one line per coin. That's it.


| id      | pred_10d | pred_30d |
| ------- | -------- | -------- |
| 0G      | 0.560    | 0.160    |
| 2Z      | 0.800    | 0.240    |
| AAVE    | 0.240    | 0.600    |
| ACE     | 0.920    | 0.640    |
| ADA     | 0.960    | 0.440    |
| AERO    | 0.680    | 0.400    |
| AIXBT   | 0.320    | 0.480    |
| ALGO    | 1.000    | 0.680    |
| ALT     | 0.400    | 0.560    |
| ANIME   | 0.280    | 0.720    |
| APE     | 0.640    | 0.280    |
| APEX    | 0.040    | 0.200    |
| APT     | 0.760    | 0.120    |
| AR      | 0.360    | 0.360    |
| ARB     | 0.600    | 0.040    |
| ASTER   | 0.880    | 1.000    |
| ATOM    | 0.480    | 0.920    |
| AVAX    | 0.160    | 0.520    |
| AVNT    | 0.720    | 0.800    |
| AXS     | 0.120    | 0.880    |
| BABY    | 0.080    | 0.080    |
| BANANA  | 0.840    | 0.760    |
| BCH     | 0.520    | 0.960    |
| BERA    | 0.200    | 0.840    |
| BIGTIME | 0.440    | 0.320    |




## What the numbers mean

You are **not** predicting prices. You're predicting a *finishing order* — like
guessing the results of a race before it's run. 1.000 means "I think this coin
finishes first"; the smallest value (1/n — about 0.006 with 170 coins) means
"I think this one finishes last."

Sort your own answer sheet and it turns into a league table:


| #   | id     | pred_10d | what it says                                  |
| --- | ------ | -------- | --------------------------------------------- |
| 1   | ALGO   | 1.000    | your strongest pick                         |
| 2   | ADA    | 0.960    |                                               |
| 3   | ACE    | 0.920    |                                               |
| 4   | ASTER  | 0.880    |                                               |
| 5   | BANANA | 0.840    |                                               |
| 6   | 2Z     | 0.800    |                                               |
| 7   | APT    | 0.760    |                                               |
| 8   | AVNT   | 0.720    |                                               |
| 9   | AERO   | 0.680    |                                               |
| 10  | APE    | 0.640    | top 10: the coins you believe in              |
| ⋮   | ⋮      | ⋮        | the middle: you're not sure, and that's fine   |
| 16  | ALT    | 0.400    | bottom 10: the coins you'd avoid              |
| 17  | AR     | 0.360    |                                               |
| 18  | AIXBT  | 0.320    |                                               |
| 19  | ANIME  | 0.280    |                                               |
| 20  | AAVE   | 0.240    |                                               |
| 21  | BERA   | 0.200    |                                               |
| 22  | AVAX   | 0.160    |                                               |
| 23  | AXS    | 0.120    |                                               |
| 24  | BABY   | 0.080    |                                               |
| 25  | APEX   | 0.040    | your weakest pick                          |




## Why the top and bottom matter most

CrowdCent's scoring (a metric called NDCG@40) mostly checks your **top ~40 and
bottom ~40** picks. Nail the extremes and the muddled middle barely counts.

That's because your ranking is really a **portfolio in disguise**: CrowdCent blends
everyone's answer sheets into one "meta-model" and builds an investment book from
it, buying the coins the crowd puts near the top, avoiding (or shorting) the ones
near the bottom. So read your own submission as an instruction:
*"BUY my top, SHORT my bottom, ignore my middle."*

## An analogy: the applause meter

Picture all 170 coins walking on stage, each having just given a performance. 
The judge calls them forward one at a time, and the crowd, every
participant in the challenge, cheers for each. Bitcoin steps up and the meter
reads 90 dB; another coin is met with near silence. Those cheers are our
submissions: each participant's ranking is one voice in the room, and the
combined volume for each coin is the meta-model.

The loudest coins advance to the next stage, the CrowdCent portfolio, as its
long positions. The coins that drew near silence advance too, but on the other
side of the book: they are the shorts. The middle of the applause range simply
stays in its seat. A submission, then, is not a forecast sheet, it is one
audience member's cheer, coin by coin, and the portfolio is built from what the
whole room agreed on loudest and quietest.
