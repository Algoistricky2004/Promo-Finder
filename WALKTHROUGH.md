# 3-minute walkthrough: what to show and what to say (about 420 words, read it in your own voice)

**0:00 - show the PDF title or the GitHub repo**
Hi, I'm Chirag. I picked Problem 1: finding promos. Today an editor watches every episode to pick 20 to 60 second clips. I built a tool that reads the screenplay and gives back those clips, with start and end times, the hook, the cliffhanger, a one-line reason and a rank.

**0:20 - show the README pipeline section**
How it works. Every line in the screenplay has a timestamp, so first the code reads that. Then the code builds every possible 20 to 60 second clip, never cutting a spoken line in half, and keeps the best few per episode. Then an AI model picks and scores them, and one more call compares the top 40 against each other. Finally the code checks everything again: real timestamps, 20 to 60 seconds, a hook in the first 3 seconds, and no invented quotes.

**0:55 - show the Colab notebook, cells 5 and 6 output**
Here it is on Colab. I upload the scripts, add my OpenAI key and run all 50 episodes. That is 51 calls and a few cents. The result: 90 promos across all 50 episodes, every one valid, and 20 near-duplicates removed by the code. The model is one setting, so Kuku can plug in a stronger one.

**1:30 - show PROMOS.md in the reference folder, rank 1**
This is the clip I would post first, from Episode 3. She is dragged out of her mother's church, a stranger steps in, he kneels with a ring, and the clip cuts before she answers.

**1:55 - show live_vs_reference.md**
Now where it fails. I compared the model with a reference ranking that Claude made by reading all 50 scripts, because I had no API key at that time. 16 of 34 reference promos match, and the extra comparison step helped a little: top-10 overlap went from 1 to 3. But the model still never says "no promo": it found one in all 50 episodes, even quiet ones. It gives 63 of 90 clips 5 out of 5 for cliffhanger, and the extra step dropped nothing. So its ranking is not reliable yet. My first attempt also failed: in free mode the model gave 2 second fragments, so I made a safer default where the code builds the clips and the model only picks.

**2:45 - show the tests output**
The code has 29 tests, and it already caught real mistakes of mine. Next I would force the model to say no, test a stronger model, and log editor accept or reject to tune it. Thanks for watching.
