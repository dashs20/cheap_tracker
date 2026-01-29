cheap_tracker is currently in its infancy, BUT, everything you need to get a tracking system of your own working is in this repo. Here are some instructions.

WHAT WILL YOU NEED?
- 2 cameras
- some retro-reflective tape
- bright LEDs
- a computer
- probably some USB extension cables

The goal is to make it super easy for OpenCV to pick out your tracking point. With retro-reflective tape, LEDs, and a dark room, you can get rid of most of the stuff you DON'T want with only contrast tweaking (which is covered later in this readme).

Mount the cameras far apart, and in different orientations. Of course - for this to work, you're going to need them both to see your tracking point, so keep that in mind. **The exact positioning and orientation doesn't matter; cheap_tracker can figure out where you put your cameras for you!** But, it needs help. So, without further ado, let's begin.

0. Tuning your camera settings

With your cameras mounted, open up tuner.py. Slide around the sliders until it's picking out your tracking points nicely. There's no right answer - it's a bit of an art.

1. Localizing your cameras

To localize your cameras, they need to BOTH SEE 4 POINTS OF KNOWN LOCATION simultaneousy. So, you're going to need to build a "device." It does not matter how the points are arranged; all that matters is you can very precisely measure/determine the dimensions of your apparatus. You will then place this "thing" in-view of both of your cameras. In **localize.py**, make sure to enter the following information:
a. camera resolution
b. FOV (this is IMPORTANT!!)
c. the openCV parameters (which you can tune with the help of **tuner.py** - see "step 0" above)
d. GROUND_TRUTH_POINTS. This is where you enter the coordinates of your 4 points. The order matters!

Once you've got all those settings dialed in, run the localization script. You should see a view from one of your cameras. Follow the instructions in the terminal, and hit space, then click points 1-4 IN ORDER. Do the same for camera 2, making sure to follow the same order.

If you do all this right, it'll spit out the correct position and orientation (in the form of a vector and a matrix) for each camera. Sanity check the positions. The orientations are a bit tougher, but the positions should be clearly right or wrong. I.E. if you mounted your camera up high and Z is -33, something's cooked

2. Running the tracker

Take those camera parameters, and paste them in the relevant spots in example.py (under the CAM portion near the top). Make sure the FOV, aspect ratio, and resolution match your localizer script! Now, hit run, and a GUI will pop up showing the points the cameras see. Also, the terminal will spit out the coordinates of up to 2 points (limited to 2 for verbosity - it can mathematically handle AS MANY AS YOU WANT)!

IF YOU HAVE ANY QUESTIONS, DM ME on instagram (https://www.instagram.com/quantum_projects_yt?igsh=b2JpcWs5ZWpoZXln) If you have code issues, go ahead and submit a github issue. I will try to address as many as I can!
