# NeuralEdit
Scene cuts guided by VGG19

## process.py
Reads a directory full of video files, it uses the bottom layers of VGG19 to detect image simillarity between frames and drops any sections that don't seem to be updating.
The features output by VGG19's fc2 dense layer and the thumbnails at changed frames are then saved.

## scan.py
Loads the metadata generated by process.py, picks a random start frame and then uses the distance to the other frame features to provide candidate follow on frames.
Once a frame is selected it is pushed to the end of a saved clip sequence for later rendering out.

### Very basic rough developer-UI for scan.py:
  Click to select the follow on frame.
  Q to exit
  W to write out current sequence
  A to skip forward to the previous meaningful frame prior to the selected base clip's end time
  D to skip forward to the previous meaningful frame next to the selected base clip's end time
  Z and X add or remove a small extension to the time the  selected base clip's end time (no update as those frames aren't saved)
  F set flip mode, to push the next clicked thumb to the image sequence but flip it horizontally.

