# 3-minute walkthrough: what to show and what to say (about 400 words, read it in your own voice)

**0:00 - show the PDF title or the GitHub repo**
Hi, I'm Chirag. I picked Problem 1: finding promos. Today an editor watches every episode to pick 20 to 60 second clips. I built a tool that reads the screenplay and gives back those clips, with start and end times, the hook, the cliffhanger, a one-line reason and a rank.

**0:20 - show the README pipeline section**
How it works. Every line in the screenplay has a timestamp, so first the code reads that. Then the code builds every possible 20 to 60 second clip, never cutting a spoken line in half, and keeps the best few per episode. Then an AI model reads those clips and picks and scores them. Finally the code checks everything again: real timestamps, 20 to 60 seconds, a hook in the first 3 seconds, and no invented quotes.

**0:50 - show the Colab notebook: cells 5 and 6 output**
Here it is on Colab. I upload the scripts, add my OpenAI key and run all 50 episodes. That is 50 calls and a few cents. The result: 87 promos across all 50 episodes, every one valid, and 18 near-duplicates removed by the code. The model is one setting, so Kuku can plug in a stronger one.

**1:25 - show PROMOS.md in the reference folder, rank 1**
This is the clip I would post first, from Episode 3. She is dragged out of her mother's church, a stranger steps in, he kneels with a ring, and the clip cuts before she answers.

**1:55 - show live_vs_reference.md**
Now where it fails. I compared the model with a reference ranking that Claude made by reading all 50 scripts, because I had no API key at that time. Only 15 of 34 reference promos match. The model never says "no promo": it found one in all 50 episodes, even the quiet ones. And it gives almost every clip 5 out of 5 for cliffhanger, 64 out of 87, even a finished speech. So its ranking is not reliable yet. My first attempt also failed: in free mode the model gave 2 second fragments, so I made a safer default where the code builds the clips and the model only picks.

**2:40 - show the tests output**
Next, I would add a second step where the model compares the top 30 clips against each other, and log editor accept or reject to tune the weights. The code has 29 tests, and it already caught real mistakes of mine. Thanks for watching.
