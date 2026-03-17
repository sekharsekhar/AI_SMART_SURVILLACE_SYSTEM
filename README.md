# AI-Based Video Surveillance System
## Real-Time Abnormal Activity Detection

---

**Developer:** Krishna Nand Pathak  
**Project Type:** Academic Project  
**Last Updated:** February 2026

---

## Hello There! Welcome to My Project 👋

Hey! I'm **Krishna Nand Pathak**, and I'm really excited that you're here to learn about this project. 

Let me be honest with you - when I first started learning about AI and computer vision, I found most tutorials and documentation really confusing. They used big words, assumed I knew things I didn't, and made everything feel way more complicated than it needed to be.

So I promised myself: when I build something, I'll explain it the way I wished someone had explained it to me.

**This guide is written for complete beginners.** Even if you've never written a line of code before, I want you to understand what this project does and how it works. We'll take it slow, step by step.

Ready? Let's go! ☕

---

## Table of Contents

1. [So What Does This Thing Actually Do?](#so-what-does-this-thing-actually-do)
2. [Why Did I Build This?](#why-did-i-build-this)
3. [The Big Picture - How It All Works](#the-big-picture---how-it-all-works)
4. [All The Features Explained](#all-the-features-explained)
5. [The Building Blocks (Packages We Use)](#the-building-blocks-packages-we-use)
6. [Project Files - What's What](#project-files---whats-what)
7. [How Each Part Works](#how-each-part-works)
8. [Setting Things Up](#setting-things-up)
9. [How To Use It](#how-to-use-it)
10. [When Things Go Wrong](#when-things-go-wrong)

---

## So What Does This Thing Actually Do?

Okay, imagine you have a security camera at your shop. Normally, you'd have to sit and watch that camera feed ALL day to make sure nothing bad happens. That's exhausting, right? And honestly, after watching for 30 minutes, your brain starts zoning out.

**This project is like hiring a really smart assistant who never gets tired.**

You point it at a video (or a live camera), and it automatically watches for:

- **Fights and Violence** - If people start hitting each other
- **Car Accidents** - If vehicles crash
- **Intrusion** - If someone enters an area they shouldn't be in
- **Loitering** - If someone is hanging around suspiciously for too long
- **Running** - If someone suddenly starts running (which could mean something's wrong)
- **Crowd Problems** - If too many people gather, or if everyone suddenly runs away (panic)

And here's the cool part: **when it sees something wrong, it sends you an email right away with a picture of what happened!**

So you don't have to watch the camera. You can do other things, and the system will tap your shoulder (via email) only when something needs your attention.

Pretty neat, huh?

---

## Why Did I Build This?

Let me tell you a real problem.

Security guards have a tough job. Studies have shown that when a person watches video screens for more than 20 minutes, their attention drops significantly. They start missing things. It's not their fault - that's just how human brains work!

Now imagine a security room with 20 camera feeds. One person is supposed to watch all of them. That's impossible to do well.

**So I thought: what if we let the computer do the boring watching part?**

Computers don't get tired. They don't get bored. They can watch 100 cameras at the same time and never lose focus.

That's what this project does - it uses artificial intelligence to automatically spot problems, so humans can focus on actually solving those problems instead of staring at screens all day.

---

## The Big Picture - How It All Works

Before we dive into details, let me give you a bird's eye view of how everything fits together. Think of it like a factory assembly line:

```
Step 1: VIDEO COMES IN
   ↓
   You either upload a video file OR use your computer's camera
   ↓
Step 2: COMPUTER LOOKS AT EACH FRAME
   ↓
   A video is just a bunch of pictures shown really fast (like a flipbook).
   The computer looks at each picture one by one.
   ↓
Step 3: FIND THE PEOPLE
   ↓
   First, we ask: "Where are the people in this picture?"
   We use something called YOLO (don't worry, I'll explain) to draw 
   boxes around every person it finds.
   ↓
Step 4: TRACK THE PEOPLE
   ↓
   We give each person a number (ID). So "Person 1" in frame 1 
   is still "Person 1" in frame 2, even if they moved.
   This lets us follow their movement over time.
   ↓
Step 5: CHECK FOR BAD STUFF
   ↓
   Now we ask several questions:
   - "Does this look like a fight?" (Violence check)
   - "Is anyone in a forbidden area?" (Intrusion check)
   - "Has anyone been standing still too long?" (Loitering check)
   - "Is anyone running?" (Running check)
   - "Are there too many people?" (Crowd check)
   ↓
Step 6: ALERT IF NEEDED
   ↓
   If any answer is YES, we:
   - Save it to a database (like a diary of events)
   - Send an email to warn someone
   - Show a warning on the video
   ↓
Step 7: SHOW THE RESULT
   ↓
   The video plays in your browser with colored boxes around 
   people and warnings when something is detected.
```

That's it! That's the whole thing. The rest is just details about how each step works.

---

## All The Features Explained

Let me explain each detection feature in simple terms:

### 1. Violence Detection 🥊

**What it does:** Spots fights, assaults, and physical attacks

**How it works (in simple words):**

Imagine you show a photo to a very smart friend and ask "Does this look like a fight?" Your friend has seen millions of photos of fights and normal activities, so they can tell the difference.

That's exactly what we do! We use something called **CLIP** (made by OpenAI) which has "seen" millions of images. We show it each video frame and ask "Hey, does this look like any of these: a fight, an assault, a punch, a kick?" 

If CLIP says "Yeah, that looks like a fight, I'm 85% sure", we trigger an alert.

### 2. Accident Detection 🚗

**What it does:** Spots car crashes and vehicle collisions

**How it works:**

Same idea as violence detection! We ask CLIP: "Does this look like a car crash? A vehicle collision? A road accident?" If it says yes with enough confidence, we alert you.

### 3. Intrusion Detection 🚷

**What it does:** Spots when someone enters a "no-go" zone

**How it works (in simple words):**

You know those invisible fences for dogs? Same idea!

You draw a shape on the screen and say "Nobody should be inside this area." The system remembers that shape. Then, for every person it detects, it checks: "Is this person's position inside the forbidden shape?" If yes, that's an intrusion alert!

It's like drawing a line in the sand and watching if anyone crosses it.

### 4. Loitering Detection ⏱️

**What it does:** Spots when someone hangs around in one place for too long

**How it works (in simple words):**

Imagine someone standing outside a jewelry shop for 10 minutes, just... standing there. That's suspicious, right?

The system keeps track of how long each person has been in roughly the same spot. If Person #5 has been within a small area for more than 60 seconds, and they haven't really moved much, that's loitering!

It's like having a timer that starts when you stop moving.

### 5. Running Detection 🏃

**What it does:** Spots when someone is running (which might mean something's wrong)

**How it works (in simple words):**

Remember how we track each person's position over time? Well, if Person #3 was at position A in frame 1, and then at position B in frame 2, and those positions are far apart... they're moving fast!

Think of it like this: if you're at the door in the first photo, and then you're at the window in the next photo taken just 1 second later, you must have run!

The system calculates how fast each person is moving. If someone is moving faster than normal walking speed, that's running.

### 6. Crowd Anomaly Detection 👥

**What it does:** Spots unusual crowd behavior

**How it works (in simple words):**

This one checks several things:

1. **Too many people:** If suddenly there are 20 people where there used to be 5, something's happening. Maybe a flash mob? A protest? An emergency?

2. **Sudden gathering:** If the number of people is rapidly increasing, people are coming together for some reason.

3. **Sudden dispersal:** If everyone suddenly leaves or runs away, that might mean panic!

4. **Stampede:** If everyone is moving fast in the same direction, that could be dangerous.

It's like noticing when a crowd forms at an accident scene, or when everyone runs away from danger.

---

## The Building Blocks (Packages We Use)

When you build something in Python, you don't write everything from scratch. That would take years! Instead, you use "packages" - code that other smart people wrote and shared freely for everyone to use.

Think of it like cooking: you don't grow your own tomatoes, make your own cheese, and mill your own flour every time. You use ingredients that others already prepared.

Here are the main ingredients (packages) we use:

### For the Smart Brain (AI Stuff)

| Package | What It Is (In Simple Words) |
|---------|------------------------------|
| **torch** | This is PyTorch - think of it as the engine that powers our AI. Just like a car needs an engine to run, our AI needs PyTorch. |
| **clip** | This is the "smart friend" who can look at any image and tell you what's happening in it. Made by OpenAI (the ChatGPT people). |
| **ultralytics** | This contains YOLO - our super-fast "Where's Waldo" finder. It can spot all the people in a picture in less than a second. |

### For Working With Videos and Images

| Package | What It Is (In Simple Words) |
|---------|------------------------------|
| **opencv-python** | This is like Photoshop for code. It lets us open videos, grab individual frames, draw boxes on them, and show the results. |
| **Pillow** | Helps us work with images in different formats. Like a translator between image types. |
| **numpy** | Images are secretly just tables of numbers! This package helps us do math on those numbers really fast. |

### For the Website Part

| Package | What It Is (In Simple Words) |
|---------|------------------------------|
| **flask** | This is what turns our Python code into a website. Without it, you'd have to run everything in a command line (boring!). |
| **werkzeug** | A helper for Flask. Makes sure file uploads are safe so nobody can upload a virus pretending to be a video. |

### For Remembering Things

| Package | What It Is (In Simple Words) |
|---------|------------------------------|
| **sqlalchemy** | This lets us save data to a database using simple Python code. It's like having a really organized notebook that never loses anything. |

### For Math

| Package | What It Is (In Simple Words) |
|---------|------------------------------|
| **scipy** | We use this for calculating distances. Like, "how far did this person move between two frames?" |

### For Settings

| Package | What It Is (In Simple Words) |
|---------|------------------------------|
| **pyyaml** | Our settings are stored in a special file called settings.yaml. This package lets us read that file. |

---

## Project Files - What's What

Let me walk you through every file and folder. I'll tell you what each one does in plain, simple language.

```
📁 Violence-Detection-Opencv-Videos-main/
│
├── 📄 app.py                 ← THE MAIN FILE! This is where everything starts.
│                               When you run "python app.py", this is what runs.
│                               Think of it as the boss who manages everyone else.
│
├── 📄 model.py               ← Contains the CLIP "smart brain".
│                               This file knows how to ask CLIP questions
│                               like "does this look like a fight?"
│
├── 📄 settings.yaml          ← THE SETTINGS FILE. Like a control panel.
│                               Want to change your email? Change it here.
│                               Want detection to be more sensitive? Change it here.
│                               All the adjustable knobs are in this file.
│
├── 📄 requirements.txt       ← Shopping list of packages we need.
│                               When you run "pip install -r requirements.txt",
│                               it reads this list and installs everything.
│
├── 📄 README.md              ← This file you're reading right now!
│
├── 📁 detectors/             ← ALL THE DETECTION BRAINS LIVE HERE
│   │
│   ├── 📄 yolo_detector.py   ← The "people finder". 
│   │                           Its job: look at a picture, find all the people.
│   │
│   ├── 📄 violence_detector.py ← The "fight detector". 
│   │                             Its job: is this a violent scene?
│   │
│   ├── 📄 intrusion_detector.py ← The "zone watcher". 
│   │                              Its job: did someone enter a forbidden area?
│   │
│   ├── 📄 loitering_detector.py ← The "standing-around detector". 
│   │                               Its job: has someone been here too long?
│   │
│   ├── 📄 running_detector.py  ← The "speed detector". 
│   │                             Its job: is someone moving too fast?
│   │
│   └── 📄 crowd_detector.py    ← The "crowd watcher". 
│                                 Its job: is the crowd behaving strangely?
│
├── 📁 tracking/              ← THE PERSON TRACKER LIVES HERE
│   │
│   └── 📄 tracker.py         ← Keeps track of who's who.
│                               "Person #5 in frame 1 is the same as 
│                               Person #5 in frame 2" - this file figures that out.
│
├── 📁 alerts/                ← THE EMAIL SYSTEM LIVES HERE
│   │
│   └── 📄 email_alert.py     ← Knows how to send emails when bad stuff happens.
│                               Like your assistant who sends you a text
│                               whenever something's wrong.
│
├── 📁 database/              ← THE MEMORY/DIARY LIVES HERE
│   │
│   ├── 📄 models.py          ← Defines what an "event" looks like.
│   │                           (What info do we save? Time? Type? How severe?)
│   │
│   └── 📄 db.py              ← Functions to save events and look them up later.
│
├── 📁 templates/             ← THE WEBSITE PAGES LIVE HERE
│   │
│   ├── 📄 index.html         ← The main page you see in your browser.
│   │                           The dashboard with stats and alerts.
│   │
│   └── 📄 single_video.html  ← The page that shows a video being analyzed.
│
├── 📁 uploaded_videos/       ← When you upload a video, it gets saved here.
│
├── 📁 snapshots/             ← When something is detected, a screenshot is
│                               saved here. These get attached to emails too.
│
└── 📁 data/                  ← Database storage
    │
    └── 📄 surveillance.db    ← The actual database file. All events are 
                                stored here. It's just one file! You could
                                copy it to a USB drive if you wanted.
```

### How Do These Files Talk to Each Other?

Let me draw a simple picture:

```
When you start the program (python app.py):

                         ┌──────────────┐
                         │    app.py    │ ← Everything starts here
                         │  (the boss)  │
                         └──────┬───────┘
                                │
           First, app.py reads the settings:
                                │
                         ┌──────▼───────┐
                         │ settings.yaml│ ← "Oh, email is ON, sensitivity is 0.2..."
                         └──────────────┘
                                │
           Then, app.py calls its team members:
                                │
      ┌───────────────┬────────┴────────┬───────────────┐
      ▼               ▼                 ▼               ▼
┌───────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
│ detectors │  │  tracking  │  │   alerts   │  │  database  │
│ "Find     │  │ "Track     │  │ "Send      │  │ "Write     │
│  problems"│  │  people"   │  │  emails"   │  │  it down"  │
└───────────┘  └────────────┘  └────────────┘  └────────────┘
                                │
           Finally, app.py shows everything in:
                                │
                         ┌──────▼───────┐
                         │  templates   │ ← The web pages you see
                         │  (HTML files)│
                         └──────────────┘
```

**Think of app.py as a manager:**
- It asks the `detectors` team: "Look at this picture, any problems?"
- It asks the `tracking` team: "Who's who in this frame?"
- It tells the `alerts` team: "Send an email about this!"
- It tells the `database` team: "Write this event in the diary!"
- It uses `templates` to show everything in a nice web page

---

## How Each Part Works

Now let me explain how each major piece works. I'll use everyday examples, not technical jargon.

### 1. How We Find People in a Picture (YOLO Detector)

**The Problem:** 
We have a picture. Where are all the people in it?

**The Old Way:**
Imagine looking at a "Where's Waldo" picture. You'd scan every little area one by one. Left side... middle... right side... This takes forever!

**The YOLO Way:**
YOLO (which stands for "You Only Look Once") is smarter. It looks at the WHOLE picture at once and instantly says "There's a person here, here, and here!"

It's like having a friend with super vision who can spot everyone immediately.

**What happens:**
```
You give YOLO: A picture

YOLO gives you back: 
  "There's a person in the top-left corner - I'm 95% sure"
  "There's another person in the middle - I'm 87% sure"
```

We ignore guesses where YOLO isn't at least 50% sure.

---

### 2. How We Track People Across Frames (Tracker)

**The Problem:** 
YOLO tells us where people are in EACH picture separately. But we need to know: "Is this the same person as before?"

**Think of it like this:**
- Photo 1: There's someone at the door
- Photo 2 (1 second later): There's someone near the door

Is that the same person, or two different people?

**The Solution:**
We use DISTANCE. If the person in photo 2 is close to where the person was in photo 1, it's probably the same person!

```
Photo 1: Person at position (100, 200) → We call them "Person #1"
Photo 2: Person at position (105, 205) → Close to before! Still "Person #1"
Photo 3: Person at position (110, 208) → Still close! Still "Person #1"
         New person at (500, 300)      → Far away! Must be "Person #2"
```

**Why this matters:**
Now we can say things like:
- "Person #1 has been standing still for 2 minutes" → Loitering!
- "Person #2 moved 100 pixels in half a second" → They're running!

---

### 3. How Violence Detection Works (CLIP)

**The Problem:** 
How do we know if a picture shows a fight?

**The Analogy:**
Imagine you have a really smart friend who has watched millions of movies and seen millions of photos. You show them a picture and ask "Does this look like a fight or just normal people?"

That's CLIP! It was trained on millions of images from the internet, each with descriptions. So it learned:
- Pictures with punching = "fight", "violence", "attack"
- Pictures with hugging = "love", "affection", "happy"
- Pictures with cars smashed = "accident", "crash", "collision"

**How we use it:**
We give CLIP a list of possible descriptions:
- "A fight on a street"
- "People fighting violently"
- "A normal peaceful scene"
- "People walking calmly"

Then we show it the current video frame and ask: "Which description fits best?"

If CLIP says "This looks like 'people fighting violently', I'm 80% sure" → We sound the alarm!

---

### 4. How Loitering Detection Works

**The Problem:** 
How do we know if someone has been standing around too long?

**The Logic (it's really simple!):**

Remember, we're tracking each person with an ID number. We know:
- When we first saw them
- Where they were
- Where they are now

So we just do simple math:
1. How long have they been visible? (Now time - First seen time)
2. How far have they moved? (Distance from starting spot to current spot)

If they've been visible for MORE than 60 seconds AND moved LESS than 50 pixels... they're loitering!

**Example:**
```
Person #5:
  - First appeared at 10:00:00, at the park bench
  - Now it's 10:02:00, still at the park bench
  
  Time hanging around: 2 minutes
  Distance moved: barely any
  
  That's loitering!
```

---

### 5. How Running Detection Works

**The Problem:** 
How do we know if someone is running?

**The Logic (also really simple!):**

Speed = Distance ÷ Time

We know:
- Where the person was in the previous frame
- Where they are now
- Time between frames (usually 1/30th of a second)

If they moved a lot in a short time = they're moving fast!

But wait - what if someone just jumped? That's one fast movement, not running.

So we check: are they moving fast for at least 3 frames in a row? If yes, they're really running!

**Example:**
```
Person #3:
  Frame 100: Position (200, 300)
  Frame 101: Position (205, 303)   ← Moved a little (walking)
  Frame 102: Position (230, 320)   ← Moved a lot (running!)
  Frame 103: Position (255, 337)   ← Still fast (still running!)
  Frame 104: Position (280, 354)   ← Still fast (definitely running!)
  
  3+ fast frames in a row = RUNNING ALERT!
```

---

### 6. How Email Alerts Work

**The Problem:** 
When something bad happens, we need to tell someone immediately.

**The Solution:**

When violence (or any problem) is detected:
1. We take a screenshot of the current frame
2. We write an email:
   - Subject: "⚠️ VIOLENCE DETECTED!"
   - Body: What happened, when, how confident we are
   - Attachment: The screenshot
3. We connect to Gmail's mail server
4. We send the email

**But there's a catch!**

Imagine a 30-second fight in a 30 fps video. That's 900 frames of violence. Without any limits, we'd send 900 emails! Your inbox would explode!

So we have a "cooldown" - after sending one violence email, we wait at least 30 seconds before sending another violence email. This way you get ONE alert per incident, not hundreds.

**About Gmail and App Passwords:**

Gmail is smart about security. It won't let random programs use your account (for good reason - what if it was a virus?).

So Google has a system called "App Passwords":
1. You go to your Google account settings
2. You create a special password JUST for this app
3. This password only works for sending emails, nothing else
4. You put this special password in settings.yaml

This way, even if someone somehow got this password, they could only send emails (not read them, not access your Drive, nothing else).

---

### 7. How the Database Works

**The Problem:** 
We want to remember what happened, even after turning off the computer.

**The Solution:**

Think of the database as a really organized notebook. Whenever something is detected, we write it down:

| When | What | How Bad | How Sure | Picture |
|------|------|---------|----------|---------|
| 10:05:23 AM | Violence | High | 85% | snapshot_1.jpg |
| 10:07:45 AM | Loitering | Medium | 92% | snapshot_2.jpg |
| 10:15:12 AM | Running | Low | 78% | snapshot_3.jpg |

This "notebook" is stored in a file called `surveillance.db`. 

**What's cool about SQLite:**
- The entire database is just ONE file
- No need to install any database software
- You could literally copy this file to a USB drive and take your event history with you!

---

## Setting Things Up

Let me walk you through setting this up, step by step.

### Step 1: Make Sure You Have Python

Open Command Prompt (search for "cmd" in Windows) and type:
```
python --version
```

If you see something like "Python 3.10.0", great! If you see an error, you need to install Python from python.org.

### Step 2: Go to the Project Folder

In Command Prompt, navigate to where you put the project:
```
cd C:\path\to\Violence-Detection-Opencv-Videos-main
```

(Replace with your actual path)

### Step 3: Create a Virtual Environment (Recommended)

This is like creating a clean room for your project:
```
python -m venv venv
```

Then activate it:
```
venv\Scripts\activate
```

You'll see `(venv)` appear at the beginning of your command line. This means you're in the clean room!

### Step 4: Install All the Packages

Now we use that shopping list:
```
pip install -r requirements.txt
```

This might take a few minutes. It's downloading all the AI stuff. Get a coffee! ☕

### Step 5: Set Up Email Alerts (Optional)

If you want to receive email alerts, open `settings.yaml` in any text editor (Notepad works fine).

Find this section:
```yaml
alerts:
  smtp:
    enabled: true
    host: "smtp.gmail.com"
    port: 587
    username: "YOUR_EMAIL@gmail.com"      ← Put YOUR Gmail address
    password: "xxxx xxxx xxxx xxxx"        ← Put your App Password
    recipients:
      - "whoever@example.com"              ← Who should get the alerts?
```

**Getting a Gmail App Password:**
1. Go to myaccount.google.com
2. Click on "Security"
3. Turn on "2-Step Verification" if it's not already on
4. Go to "App passwords" (you can search for it)
5. Create a new app password for "Mail"
6. Copy the 16-character code
7. Paste it in settings.yaml

### Step 6: Run the Program!

```
python app.py
```

You should see:
```
* Running on http://127.0.0.1:5000
```

### Step 7: Open Your Browser

Go to:
```
http://127.0.0.1:5000
```

You should see the dashboard! 🎉

---

## How To Use It

### Analyzing a Video

1. On the main page, click "Upload Video" or similar button
2. Choose a video file (MP4, AVI, MOV, etc.)
3. Click Upload/Analyze
4. Watch the video play with detection overlays
5. Look at the alerts panel for any detected issues

### Using Your Webcam

1. Click "Live Camera" or "Webcam"
2. Allow camera access when your browser asks
3. The system will analyze what your camera sees in real-time
4. Any detected problems show up as alerts

### Understanding What You See

- **Green boxes** = People detected (normal)
- **Red boxes** = Potential problem detected
- **Status panel** = Shows current detection status
- **Alerts list** = History of detected events

---

## When Things Go Wrong

Don't panic! Here are common problems and how to fix them:

### "It's running really slowly!"

The AI needs computing power. If you don't have a fancy graphics card (GPU), it will use your regular processor (CPU), which is slower.

**What you can do:**
- Be patient - it still works, just slower
- Close other programs to free up resources
- If you have an NVIDIA graphics card, look up how to install CUDA

### "Emails aren't being sent!"

Check these things:
1. In settings.yaml, is `enabled: true` under smtp?
2. Did you use an App Password (not your regular Gmail password)?
3. Is your Gmail correct? Any typos?
4. Is the recipient email correct?

### "It's detecting too many false alarms!"

The system might be too sensitive. Open settings.yaml and find:
```yaml
detection-settings:
  violence:
    threshold: 0.2
```

Try increasing this number (like 0.3 or 0.4). Higher number = less sensitive.

### "It's missing real events!"

The opposite problem - it's not sensitive enough. Try lowering the threshold (like 0.15 or 0.1).

### "Camera isn't working!"

- Is another app using your camera? (Zoom, Teams, Skype?)
- Try closing all other apps and try again
- Try a different browser

### "I'm getting weird error messages about packages!"

Try reinstalling everything:
```
pip install --upgrade -r requirements.txt
```

---

## What You've Learned Today

Congratulations! You now understand:

✅ What this surveillance system does and why it's useful  
✅ How AI can watch videos and spot problems automatically  
✅ How YOLO finds people in pictures  
✅ How tracking remembers who's who  
✅ How CLIP understands what's happening in a scene  
✅ How loitering, running, and crowd detection work  
✅ How the system sends email alerts  
✅ How to set up and run the project  

---

## Final Words

When I started this project, I didn't know half of what I know now. I made mistakes. I broke things. I googled error messages at 2 AM.

That's how learning works.

Don't be intimidated by AI or computer vision. At its heart, this is all just:
1. Look at a picture
2. Ask simple questions about it
3. Do something if the answer is concerning

That's it. The fancy AI stuff is just a smarter way of asking those questions.

If you get stuck, don't give up. Every programmer has been stuck. Every programmer has felt dumb. The difference is that good programmers keep going anyway.

**Happy learning!** 🚀

---

**Developer:** Krishna Nand Pathak  
**Project:** AI Video Surveillance System  
**Date:** February 2026

---

*Remember: The best way to learn is by doing. Don't just read this - run the code, break it, fix it, change things and see what happens. That's how you become a real developer!*
