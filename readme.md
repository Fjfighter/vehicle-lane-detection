Code Organization:
    main.py - contains the main function where input source is being processed both by the lane detection system as well as the vehicle recognition system.
    utils.py - contains helper functions to process image as well as compute final detection
    settings.py - contains some helper flags for settings used during debugging. 

How to Run:
    The system requires a sample dataset in order to run. A short sample dataset has been included in /videos directory.
    
    To run using included sample dataset, use the command:

        python main.py

    To run with a provided dataset (.mp4 video file), use the command:

        python main.py PATH_TO_DATASET