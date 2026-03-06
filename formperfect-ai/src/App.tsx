/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/// <reference types="vite/client" />

import { Pose } from '@mediapipe/pose';
import React, { useState, useEffect, useRef } from 'react';
import { GoogleGenAI } from "@google/genai";
import { 
  Camera, Activity, CheckCircle, AlertTriangle, RefreshCw, Info, 
  ChevronRight, XCircle, Calendar, Plus, Trash2, Play, ArrowLeft, 
  Dumbbell, GripVertical, BarChart2, Clock, Target, Zap, RotateCcw, 
  Minus, Timer, Save, ArrowRight, User, Settings, LogOut, Edit2
} from 'lucide-react';

// --- Helper Math Functions ---
function calculateAngle(A: any, B: any, C: any) {
  const radians = Math.atan2(C.y - B.y, C.x - B.x) - Math.atan2(A.y - B.y, A.x - B.x);
  let angle = Math.abs(radians * 180.0 / Math.PI);
  if (angle > 180.0) angle = 360 - angle;
  return angle;
}

const JOINT_INDICES: Record<string, number[]> = {
  shoulder: [11, 12],
  elbow: [13, 14],
  wrist: [15, 16],
  hip: [23, 24],
  knee: [25, 26],
  ankle: [27, 28]
};

function getBestJoint(landmarks: any[], jointName: string, preferredSide?: 'left' | 'right' | null) {
  const indices = JOINT_INDICES[jointName];
  if (!indices || !landmarks) return null;
  const left = landmarks[indices[0]];
  const right = landmarks[indices[1]];
  
  if (!left && !right) return null;
  
  if (preferredSide === 'left' && left && (left.visibility || 0) > 0.3) return { ...left, score: left.visibility || 0 };
  if (preferredSide === 'right' && right && (right.visibility || 0) > 0.3) return { ...right, score: right.visibility || 0 };

  if (!left) return { ...right, score: right.visibility || 0 };
  if (!right) return { ...left, score: left.visibility || 0 };
  
  return (left.visibility || 0) > (right.visibility || 0) 
    ? { ...left, score: left.visibility || 0 } 
    : { ...right, score: right.visibility || 0 };
}

// --- AI Form Checkers ---
const noSwingingCheck = (landmarks: any[]) => {
  const shoulder = getBestJoint(landmarks, 'shoulder');
  const hip = getBestJoint(landmarks, 'hip');
  const ankle = getBestJoint(landmarks, 'ankle');
  if (shoulder && hip && ankle && shoulder.score > 0.4 && hip.score > 0.4 && ankle.score > 0.4) {
     const bodyAngle = calculateAngle(shoulder, hip, ankle);
     if (bodyAngle < 165) return "Stop swinging your back!";
  }
  return null;
};

const chestUpCheck = (landmarks: any[]) => {
  const shoulder = getBestJoint(landmarks, 'shoulder');
  const hip = getBestJoint(landmarks, 'hip');
  const knee = getBestJoint(landmarks, 'knee');
  if (shoulder && hip && knee && shoulder.score > 0.4 && hip.score > 0.4 && knee.score > 0.4) {
     const torsoAngle = calculateAngle(shoulder, hip, knee);
     if (torsoAngle < 60) return "Keep your chest up!";
  }
  return null;
};

const noFlaringCheck = (landmarks: any[]) => {
  const shoulder = getBestJoint(landmarks, 'shoulder');
  const elbow = getBestJoint(landmarks, 'elbow');
  const hip = getBestJoint(landmarks, 'hip');
  if (shoulder && elbow && hip && shoulder.score > 0.4 && elbow.score > 0.4 && hip.score > 0.4) {
     const flareAngle = calculateAngle(hip, shoulder, elbow);
     if (flareAngle > 75) return "Tuck your elbows!";
  }
  return null;
};

const fullLockoutCheck = (landmarks: any[]) => {
  const shoulder = getBestJoint(landmarks, 'shoulder');
  const elbow = getBestJoint(landmarks, 'elbow');
  const wrist = getBestJoint(landmarks, 'wrist');
  if (shoulder && elbow && wrist && shoulder.score > 0.4 && elbow.score > 0.4 && wrist.score > 0.4) {
     const armAngle = calculateAngle(shoulder, elbow, wrist);
     if (armAngle < 160) return "Lock your arms at the top!";
  }
  return null;
};

const createSmartAnalyzer = (config: any) => {
  const { 
    joint1, joint2, joint3, 
    isPush, minRange, 
    msgDown, msgUp, msgPerfect, msgBadDepth, 
    customCheck,
    idealMin, idealMax // New: Target angles for "Exact" form
  } = config;
  return (landmarks: any[], repState: any, setReps: any, setFeedback: any, onProgress: any) => {
    // 1. Joint Persistence & Side Switching
    const indices = JOINT_INDICES[joint2];
    const j2L = indices ? landmarks[indices[0]] : null;
    const j2R = indices ? landmarks[indices[1]] : null;
    
    if (!repState.current.preferredSide) {
      if (j2L && j2R) {
        repState.current.preferredSide = (j2L.visibility || 0) > (j2R.visibility || 0) ? 'left' : 'right';
      }
    } else {
      // If preferred side becomes obscured, switch
      const currentPref = repState.current.preferredSide === 'left' ? j2L : j2R;
      const otherPref = repState.current.preferredSide === 'left' ? j2R : j2L;
      if ((currentPref?.visibility || 0) < 0.2 && (otherPref?.visibility || 0) > 0.5) {
        repState.current.preferredSide = repState.current.preferredSide === 'left' ? 'right' : 'left';
        repState.current.calibratedMin = 180; // Reset calibration on side switch
        repState.current.calibratedMax = 0;
      }
    }

    const j1 = getBestJoint(landmarks, joint1, repState.current.preferredSide);
    const j2 = getBestJoint(landmarks, joint2, repState.current.preferredSide);
    const j3 = getBestJoint(landmarks, joint3, repState.current.preferredSide);
    
    if (!j1 || !j2 || !j3 || j1.score < 0.4 || j2.score < 0.4 || j3.score < 0.4) {
      setFeedback("Body obscured. Adjust position.");
      return;
    }
    const rawAngle = calculateAngle(j1, j2, j3);
    const prevAngle = repState.current.lastAngle;
    const now = performance.now();
    
    // 2. Velocity Filtering: Ignore impossible jumps (noise)
    if (prevAngle !== null) {
      const dt = (now - repState.current.lastFrameTime) / 1000;
      const velocity = Math.abs(rawAngle - prevAngle) / dt; // deg/sec
      if (velocity > 800 && dt > 0) { // Increased threshold slightly
        repState.current.lastFrameTime = now;
        return; 
      }
    }
    repState.current.lastFrameTime = now;

    // 3. Adaptive Smoothing: More stable
    const smoothing = 0.4; 
    const angle = prevAngle !== null ? (prevAngle * (1 - smoothing) + rawAngle * smoothing) : rawAngle;
    
    if (prevAngle !== null && Math.abs(angle - prevAngle) > 0.5) {
      repState.current.lastMoveTime = now;
    }
    
    // Calibration with "Ideal" bias
    if (angle < repState.current.calibratedMin || repState.current.calibratedMin === 180) { 
      repState.current.calibratedMin = angle; 
      repState.current.lastMinTime = now; 
    }
    if (angle > repState.current.calibratedMax || repState.current.calibratedMax === 0) { 
      repState.current.calibratedMax = angle; 
      repState.current.lastMaxTime = now; 
    }
    
    // Decay calibration towards ideal if provided
    if (now - repState.current.lastMinTime > 5000) {
      if (idealMin !== undefined) repState.current.calibratedMin = repState.current.calibratedMin * 0.95 + idealMin * 0.05;
      else repState.current.calibratedMin += 0.5;
    }
    if (now - repState.current.lastMaxTime > 5000) {
      if (idealMax !== undefined) repState.current.calibratedMax = repState.current.calibratedMax * 0.95 + idealMax * 0.05;
      else repState.current.calibratedMax -= 0.5;
    }
    
    const dynamicRange = Math.max(15, repState.current.calibratedMax - repState.current.calibratedMin);
    if (dynamicRange < minRange) { setFeedback("One full rep to calibrate."); if (onProgress) onProgress(0, repState.current.phase, isPush); return; }
    
    let percent = 0;
    if (isPush) percent = ((angle - repState.current.calibratedMin) / dynamicRange) * 100;
    else percent = ((repState.current.calibratedMax - angle) / dynamicRange) * 100;
    percent = Math.max(0, Math.min(100, Math.round(percent)));
    repState.current.currentRepRom = Math.max(repState.current.currentRepRom || 0, percent);
    
    // 6. Form Scoring Logic (Stability, Tempo, ROM)
    const range = Math.abs(idealMax - idealMin);
    const currentFromMin = Math.abs(angle - idealMin);
    const romProgress = Math.min(100, Math.max(0, (currentFromMin / range) * 100));
    
    // Stability Check: Calculate variance in last 5 frames
    if (!repState.current.angleHistory) repState.current.angleHistory = [];
    repState.current.angleHistory.push(angle);
    if (repState.current.angleHistory.length > 5) repState.current.angleHistory.shift();
    const variance = repState.current.angleHistory.length > 1 
      ? Math.max(...repState.current.angleHistory) - Math.min(...repState.current.angleHistory)
      : 0;
    
    const stabilityPenalty = Math.max(0, (variance - 2) * 5); // Penalty if jitter > 2 degrees
    const currentScore = Math.max(0, romProgress - stabilityPenalty);
    
    if (onProgress) onProgress(romProgress, currentScore, repState.current.phase);
    
    let formWarning = null;
    if (customCheck) formWarning = customCheck(landmarks);
    
    const timeInPhase = now - repState.current.phaseStartTime;

    // Exact Form: Check against ideal if provided
    if (idealMin !== undefined && idealMax !== undefined) {
      const targetROM = Math.abs(idealMax - idealMin);
      const userROM = Math.abs(angle - (isPush ? idealMin : idealMax));
      const currentROMPercent = (userROM / targetROM) * 100;

      // Only warn if they are moving slowly or reversing early
      const isStalling = timeInPhase > 1200 && Math.abs(angle - (prevAngle || angle)) < 0.5;
      const isReversing = prevAngle !== null && (
        (repState.current.phase === 'up' && (isPush ? angle < prevAngle - 0.8 : angle > prevAngle + 0.8)) ||
        (repState.current.phase === 'down' && (isPush ? angle > prevAngle + 0.8 : angle < prevAngle - 0.8))
      );

      if (repState.current.phase === 'up') {
        if (currentROMPercent < 88 && (isStalling || isReversing)) {
          formWarning = isPush ? "Push higher for full lockout!" : "Curl higher! Squeeze at the top.";
        }
      } else if (repState.current.phase === 'down') {
        if (currentROMPercent > 12 && (isStalling || isReversing)) {
          formWarning = isPush ? "Lower further for full stretch!" : "Extend fully! Straighten your arm.";
        }
      }
    }

    // Stricter Partial Rep Detection
    if (repState.current.phase === 'down' && prevAngle !== null) {
      if (isPush) {
        if (angle > prevAngle + 1.5 && percent < 85) formWarning = "Don't cheat! Go deeper.";
      } else {
        if (angle < prevAngle - 1.5 && percent < 85) formWarning = "Full extension needed!";
      }
    }

    if (formWarning) setFeedback(formWarning);

    // 4. Hysteresis: Use a buffer to prevent double-triggering
    const buffer = dynamicRange * 0.07; // Stricter buffer (7%)
    const upperThreshold = repState.current.calibratedMax - buffer; 
    const lowerThreshold = repState.current.calibratedMin + buffer; 
    
    if (isPush) {
      // PUSH: Bottom (Min) -> Top (Max) -> Bottom (Min)
      if (angle > upperThreshold && repState.current.phase === 'up') {
        if (timeInPhase > 300) {
          repState.current.phase = 'down';
          repState.current.lastConcentricTime = timeInPhase;
          repState.current.phaseStartTime = now;
          setFeedback("Top reached. Now control the descent.");
        }
      } else if (angle < lowerThreshold && repState.current.phase === 'down') {
        if (timeInPhase > 300) {
          const rom = repState.current.currentRepRom;
          const eccTime = timeInPhase;
          const concTime = repState.current.lastConcentricTime || 1000;
          
          repState.current.phase = 'up';
          repState.current.phaseStartTime = now;
          repState.current.history.push({ rom, eccentricTime: eccTime, concentricTime: concTime });

          if (rom >= 92) { // Stricter ROM requirement
            repState.current.count += 1;
            setReps(repState.current.count);
            let msg = msgPerfect;
            if (eccTime < 1800) msg = "Good rep! Try to lower even slower.";
            setFeedback(msg);
          } else {
            setFeedback("Partial rep detected. Focus on full range.");
          }
          repState.current.currentRepRom = 0;
        }
      }
    } else {
      // PULL/CURL: Bottom (Max) -> Top (Min) -> Bottom (Max)
      if (angle < lowerThreshold && repState.current.phase === 'up') {
        if (timeInPhase > 300) {
          repState.current.phase = 'down';
          repState.current.lastConcentricTime = timeInPhase;
          repState.current.phaseStartTime = now;
          setFeedback("Peak contraction! Lower with control.");
        }
      } else if (angle > upperThreshold && repState.current.phase === 'down') {
        if (timeInPhase > 300) {
          const rom = repState.current.currentRepRom;
          const eccTime = timeInPhase;
          const concTime = repState.current.lastConcentricTime || 1000;
          
          repState.current.phase = 'up';
          repState.current.phaseStartTime = now;
          repState.current.history.push({ rom, eccentricTime: eccTime, concentricTime: concTime });
          
          if (rom >= 92) { // Stricter ROM requirement
            repState.current.count += 1;
            setReps(repState.current.count);
            let msg = msgPerfect;
            if (eccTime < 1800) msg = "Excellent! Keep that slow tempo.";
            setFeedback(msg);
          } else {
            setFeedback("Incomplete range. Fully extend at the bottom.");
          }
          repState.current.currentRepRom = 0;
        }
      }
    }
    repState.current.lastAngle = angle;
  };
};

const EXERCISE_DB = [
  { id: 'c1', muscleGroup: 'Chest', name: 'Push-up', isCompound: true, tips: ['Core tight', 'Break 90 deg'], analyze: createSmartAnalyzer({ joint1: 'shoulder', joint2: 'elbow', joint3: 'wrist', isPush: true, minRange: 50, idealMin: 70, idealMax: 170, msgDown: "Good depth! Push.", msgUp: "Control descent.", msgPerfect: "Perfect rep!", msgBadDepth: "Go lower!", customCheck: (l: any) => noSwingingCheck(l) || noFlaringCheck(l) }) },
  { id: 'c2', muscleGroup: 'Chest', name: 'Incline DB Press', isCompound: true, tips: ['Bench 30-45 deg', 'Tuck elbows'], analyze: createSmartAnalyzer({ joint1: 'shoulder', joint2: 'elbow', joint3: 'wrist', isPush: true, minRange: 60, idealMin: 65, idealMax: 165, msgDown: "Stretch! Press!", msgUp: "Lower slowly.", msgPerfect: "Great squeeze!", msgBadDepth: "Lower more.", customCheck: noFlaringCheck }) },
  { id: 'c3', muscleGroup: 'Chest', name: 'Chest Fly', isCompound: false, tips: ['Slight elbow bend', 'Hug a barrel'], analyze: createSmartAnalyzer({ joint1: 'shoulder', joint2: 'elbow', joint3: 'wrist', isPush: true, minRange: 30, idealMin: 110, idealMax: 170, msgDown: "Stretch! Fly!", msgUp: "Open slowly.", msgPerfect: "Great squeeze!", msgBadDepth: "Wider stretch!" }) },
  { id: 'bk1', muscleGroup: 'Back', name: 'Lat Pulldown', isCompound: true, tips: ['Pull to chest', 'No momentum'], analyze: createSmartAnalyzer({ joint1: 'shoulder', joint2: 'elbow', joint3: 'wrist', isPush: false, minRange: 60, idealMin: 45, idealMax: 170, msgDown: "Stretch! Pull down!", msgUp: "Control back up.", msgPerfect: "Perfect pull!", msgBadDepth: "Pull to chest." }) },
  { id: 'bi1', muscleGroup: 'Biceps', name: 'Bicep Curls', isCompound: false, tips: ['Pin elbows', 'No swing'], analyze: createSmartAnalyzer({ joint1: 'shoulder', joint2: 'elbow', joint3: 'wrist', isPush: false, minRange: 60, idealMin: 35, idealMax: 170, msgDown: "Extended. Curl!", msgUp: "Lower slowly.", msgPerfect: "Strict curl!", msgBadDepth: "Full stretch!", customCheck: noSwingingCheck }) },
  { id: 'bi2', muscleGroup: 'Biceps', name: 'Preacher Curls', isCompound: false, tips: ['Snug on pad', 'Full stretch'], analyze: createSmartAnalyzer({ joint1: 'shoulder', joint2: 'elbow', joint3: 'wrist', isPush: false, minRange: 60, idealMin: 40, idealMax: 165, msgDown: "Stretch! Curl!", msgUp: "Resist down.", msgPerfect: "Peak tension!", msgBadDepth: "Full extension!" }) },
  { id: 'tr1', muscleGroup: 'Triceps', name: 'Tricep Pushdowns', isCompound: false, tips: ['Pin elbows', 'Lock out'], analyze: createSmartAnalyzer({ joint1: 'shoulder', joint2: 'elbow', joint3: 'wrist', isPush: true, minRange: 60, idealMin: 45, idealMax: 175, msgDown: "Stretch! Push!", msgUp: "Control up.", msgPerfect: "Solid lockout!", msgBadDepth: "Lock your arms.", customCheck: fullLockoutCheck }) },
  { id: 'sh1', muscleGroup: 'Shoulders', name: 'Shoulder Press', isCompound: true, tips: ['Press overhead', 'Ear level'], analyze: createSmartAnalyzer({ joint1: 'shoulder', joint2: 'elbow', joint3: 'wrist', isPush: true, minRange: 60, idealMin: 60, idealMax: 175, msgDown: "Ear level. Press!", msgUp: "Lower slowly.", msgPerfect: "Solid press!", msgBadDepth: "Go deeper.", customCheck: (l: any) => noFlaringCheck(l) || fullLockoutCheck(l) }) },
  { id: 'lg1', muscleGroup: 'Legs', name: 'Squat', isCompound: true, tips: ['Below knees', 'Chest up'], analyze: createSmartAnalyzer({ joint1: 'hip', joint2: 'knee', joint3: 'ankle', isPush: true, minRange: 60, idealMin: 60, idealMax: 170, msgDown: "Depth! Drive up.", msgUp: "Control down.", msgPerfect: "Perfect depth!", msgBadDepth: "Squat deeper!", customCheck: chestUpCheck }) },
  { id: 'lg2', muscleGroup: 'Legs', name: 'Leg Extensions', isCompound: false, tips: ['Hold squeeze', 'Full range'], analyze: createSmartAnalyzer({ joint1: 'hip', joint2: 'knee', joint3: 'ankle', isPush: false, minRange: 50, idealMin: 85, idealMax: 175, msgDown: "Stretch! Kick up!", msgUp: "Lower slowly.", msgPerfect: "Great quad!", msgBadDepth: "Extend fully!" }) },
];

const MUSCLE_GROUPS = ['Chest', 'Back', 'Legs', 'Shoulders', 'Biceps', 'Triceps'];
const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
const AVATARS = [
  'https://api.dicebear.com/7.x/avataaars/svg?seed=Felix',
  'https://api.dicebear.com/7.x/avataaars/svg?seed=Aneka',
  'https://api.dicebear.com/7.x/avataaars/svg?seed=Jack',
  'https://api.dicebear.com/7.x/avataaars/svg?seed=Milo'
];

export default function App() {
  // Simple Auth State
  const [activeAccount, setActiveAccount] = useState<any>(null); 
  const [authMode, setAuthMode] = useState('login'); 
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [profile, setProfile] = useState({ name: 'Athlete', bio: 'Training for growth', pfp: AVATARS[0] });
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [authError, setAuthError] = useState('');
  const [authSuccess, setAuthSuccess] = useState('');
  const [isAuthLoading, setIsAuthLoading] = useState(true);

  // Gemini AI for recognition
  const [isDetecting, setIsDetecting] = useState(false);
  const aiRef = useRef<any>(null);

  // App Navigation & logic state
  const [view, setView] = useState('schedule'); 
  const [isLoaded, setIsLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [draggedIndex, setDraggedIndex] = useState<number | null>(null);
  const [showSaved, setShowSaved] = useState(false);
  const [selectedDay, setSelectedDay] = useState('Mon');
  const [routines, setRoutines] = useState<any>({ Mon: ['c2', 'c1', 'c3', 'bi1', 'bi2'], Tue: ['bk1', 'tr1', 'sh1'], Wed: ['lg1', 'lg2'], Thu: [], Fri: [], Sat: [], Sun: [] });
  const [workoutHistory, setWorkoutHistory] = useState<any[]>([]);
  
  const [activeExercise, setActiveExercise] = useState<any>(null);
  const [targetSets, setTargetSets] = useState(3);
  const [includeWarmup, setIncludeWarmup] = useState(true);
  const [currentSet, setCurrentSet] = useState(1);
  const [isSetupPhase, setIsSetupPhase] = useState(true);
  const [restTimeLeft, setRestTimeLeft] = useState(0);
  const [cameraActive, setCameraActive] = useState(false);
  const [reps, setReps] = useState(0);
  const [formScore, setFormScore] = useState(0);
  const [formQuality, setFormQuality] = useState<'Excellent' | 'Good' | 'Fair' | 'Poor'>('Excellent');
  const [lastRepScore, setLastRepScore] = useState<number | null>(null);
  const [feedback, setFeedback] = useState("AI is loading...");
  const [workoutStats, setWorkoutStats] = useState<any>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const poseRef = useRef<any>(null);
  const requestRef = useRef<number | null>(null);
  const cameraActiveRef = useRef(false);
  const onResultsRef = useRef<any>(null);
  const activeExerciseRef = useRef<any>(null);
  const repState = useRef<any>({ 
    phase: 'up', 
    count: 0, 
    calibratedMin: 180, 
    calibratedMax: 0, 
    lastAngle: null, 
    currentRepRom: 0, 
    phaseStartTime: 0, 
    lastEccentricTime: 0, 
    history: [], 
    lastMinTime: 0, 
    lastMaxTime: 0, 
    lastMoveTime: 0,
    lastFrameTime: 0,
    preferredSide: null
  });
  const progressFillRef = useRef<HTMLDivElement>(null);
  const progressTextRef = useRef<HTMLDivElement>(null);
  const progressLabelRef = useRef<HTMLDivElement>(null);

  useEffect(() => { activeExerciseRef.current = activeExercise; }, [activeExercise]);

  const onResults = (results: any) => {
    if (!canvasRef.current || !activeExerciseRef.current) return;
    
    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;
    
    ctx.save();
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    
    if (!results.poseLandmarks) {
      setFeedback("Searching for body...");
      ctx.restore();
      return;
    }
    
    // Draw landmarks
    if (results.poseLandmarks) {
      drawSkeleton(results.poseLandmarks, ctx);
    }
    
    // Status indicator on canvas
    ctx.fillStyle = '#8B5CF6';
    ctx.beginPath();
    ctx.arc(20, 20, 5, 0, 2 * Math.PI);
    ctx.fill();
    ctx.font = '10px sans-serif';
    ctx.fillText('AI TRACKING', 30, 24);

    // Analyze
    if (activeExerciseRef.current && results.poseLandmarks) {
      activeExerciseRef.current.analyze(results.poseLandmarks, repState, setReps, setFeedback, (p: number, score: number, ph: string) => {
        if (progressFillRef.current && progressTextRef.current && progressLabelRef.current) {
          setFormScore(Math.round(score));
          progressFillRef.current.style.width = `${p}%`;
          progressTextRef.current.innerText = `${Math.round(p)}%`;
          const isCon = ph === 'down';
          progressFillRef.current.style.backgroundColor = isCon ? '#8B5CF6' : '#6366F1';
          progressTextRef.current.style.color = isCon ? '#8B5CF6' : '#6366F1';
          progressLabelRef.current.innerText = isCon ? 'FLEX' : 'STRETCH';
        }
      });
    }

    const now = performance.now();
    if (repState.current.count > 0 && now - repState.current.lastMoveTime > 8000 && repState.current.phase === 'up') {
      stopCameraAndAnalyze();
    }
    
    ctx.restore();
  };

  useEffect(() => { onResultsRef.current = onResults; }, [onResults]);

  // --- Step 1: Initial Auth & Load MediaPipe ---
  useEffect(() => {
    const initApp = async () => {
      // Simple local auth check
      const savedUser = localStorage.getItem('formperfect_user');
      if (savedUser) {
        const userData = JSON.parse(savedUser);
        setActiveAccount(userData);
        if (userData.profile) setProfile(userData.profile);
        if (userData.routines) setRoutines(userData.routines);
        if (userData.history) setWorkoutHistory(userData.history);
      }
      setIsAuthLoading(false);

      try {
        const VERSION = '0.5.1675469404';
        const pose = new Pose({
          locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose@${VERSION}/${file}`
        });

        pose.setOptions({
          modelComplexity: 1,
          smoothLandmarks: true,
          enableSegmentation: false,
          smoothSegmentation: false,
          minDetectionConfidence: 0.3,
          minTrackingConfidence: 0.3
        });

        pose.onResults((results: any) => {
          if (onResultsRef.current) {
            try {
              onResultsRef.current(results);
            } catch (err) {
              console.error("Error in onResults:", err);
            }
          }
        });

        poseRef.current = pose;
        setIsLoaded(true);
        setFeedback("AI Ready. Select an exercise.");
      } catch (e) { 
        console.error("MediaPipe Load Error:", e);
        setError("AI load failed. Please refresh."); 
      }
    };
    initApp();
  }, []);

  // Timer Effect
  useEffect(() => {
    if (view !== 'rest') return;
    const interval = setInterval(() => { setRestTimeLeft(v => { if (v <= 1) { clearInterval(interval); return 0; } return v - 1; }); }, 1000);
    return () => clearInterval(interval);
  }, [view]);

  // --- Auth Handler ---
  const hashPassword = async (password: string) => {
    const encoder = new TextEncoder();
    const data = encoder.encode(password);
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  };

  const handleAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    setAuthError('');
    setAuthSuccess('');
    
    const hashedPassword = await hashPassword(password);
    
    if (authMode === 'signup') {
      const users = JSON.parse(localStorage.getItem('formperfect_users') || '{}');
      if (users[username]) {
        setAuthError("Username already exists.");
      } else {
        users[username] = { username, password: hashedPassword, profile, routines };
        localStorage.setItem('formperfect_users', JSON.stringify(users));
        setAuthSuccess("Sign up successful! Please log in.");
        setAuthMode('login');
        setPassword('');
      }
    } else {
      const users = JSON.parse(localStorage.getItem('formperfect_users') || '{}');
      const user = users[username];
      
      // Check if password matches (handling both plain text for migration and hashed)
      const isMatch = user && (user.password === hashedPassword || user.password === password);
      
      if (!user || !isMatch) {
        setAuthError("Invalid username or password.");
      } else {
        // Migrate to hashed if it was plain text
        if (user.password === password) {
          user.password = hashedPassword;
          users[username] = user;
          localStorage.setItem('formperfect_users', JSON.stringify(users));
        }
        
        localStorage.setItem('formperfect_user', JSON.stringify(user));
        setActiveAccount(user);
        if (user.profile) setProfile(user.profile);
        if (user.routines) setRoutines(user.routines);
      }
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('formperfect_user');
    setActiveAccount(null);
    setUsername('');
    setPassword('');
    setProfile({ name: 'Athlete', bio: 'Training for growth', pfp: AVATARS[0] });
    setView('schedule');
  };

  const handleSaveAll = () => {
    setShowSaved(true);
    const users = JSON.parse(localStorage.getItem('formperfect_users') || '{}');
    if (activeAccount) {
      const updatedUser = { ...activeAccount, profile, routines, history: workoutHistory };
      users[activeAccount.username] = updatedUser;
      localStorage.setItem('formperfect_users', JSON.stringify(users));
      localStorage.setItem('formperfect_user', JSON.stringify(updatedUser));
      setActiveAccount(updatedUser);
    }
    setTimeout(() => setShowSaved(false), 2000);
  };

  // Auto-save effect
  useEffect(() => {
    if (activeAccount && isLoaded) {
      const users = JSON.parse(localStorage.getItem('formperfect_users') || '{}');
      const updatedUser = { ...activeAccount, profile, routines, history: workoutHistory };
      users[activeAccount.username] = updatedUser;
      localStorage.setItem('formperfect_users', JSON.stringify(users));
      localStorage.setItem('formperfect_user', JSON.stringify(updatedUser));
    }
  }, [profile, routines, workoutHistory]);

  const handleUpdateProfile = (e: React.FormEvent) => {
    e.preventDefault();
    setIsProfileOpen(false);
    handleSaveAll();
  };

  // --- Routine Handlers ---
  const startWorkout = () => { 
    const day = routines[selectedDay]; 
    if (!day || day.length === 0) return alert("Add exercises first!"); 
    switchExercise(day[0]); 
  };

  const switchExercise = (id: string) => {
    const ex = EXERCISE_DB.find(e => e.id === id);
    if (!ex) return;
    setActiveExercise(ex);
    setTargetSets(3); setIncludeWarmup(true); setCurrentSet(1); setIsSetupPhase(true); setReps(0);
    const now = performance.now();
    repState.current = { phase: 'up', count: 0, calibratedMin: 180, calibratedMax: 0, lastAngle: null, currentRepRom: 0, phaseStartTime: now, lastEccentricTime: 0, history: [], lastMinTime: now, lastMaxTime: now, lastMoveTime: now };
    setView('workout');
  };

  const handleStartSetPhase = () => { setIsSetupPhase(false); setCurrentSet(includeWarmup ? 0 : 1); setReps(0); startCamera(); };
  const resetReps = () => { 
    setReps(0); 
    const now = performance.now(); 
    repState.current = { 
      ...repState.current, 
      count: 0, 
      calibratedMin: 180, 
      calibratedMax: 0, 
      currentRepRom: 0, 
      phaseStartTime: now, 
      history: [], 
      lastMoveTime: now,
      lastFrameTime: now,
      preferredSide: null,
      lastAngle: null
    }; 
    setFeedback("Recalibrating..."); 
  };
  
  const detectExercise = async () => {
    if (!videoRef.current || !canvasRef.current) return;
    setIsDetecting(true);
    setFeedback("AI scanning exercise...");

    try {
      // Capture current frame
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = videoRef.current.videoWidth;
      tempCanvas.height = videoRef.current.videoHeight;
      const ctx = tempCanvas.getContext('2d');
      if (!ctx) return;
      ctx.drawImage(videoRef.current, 0, 0);
      const base64Image = tempCanvas.toDataURL('image/jpeg').split(',')[1];

      if (!aiRef.current) {
        aiRef.current = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });
      }

      const model = "gemini-3-flash-preview";
      const exerciseNames = EXERCISE_DB.map(e => e.name).join(', ');
      
      const result = await aiRef.current.models.generateContent({
        model,
        contents: [{
          parts: [
            { text: `Identify exactly which gym exercise is being performed in this image. Choose ONLY from this list: ${exerciseNames}. Return ONLY the name of the exercise.` },
            { inlineData: { mimeType: "image/jpeg", data: base64Image } }
          ]
        }]
      });

      const detectedName = result.text?.trim();
      const exercise = EXERCISE_DB.find(e => detectedName?.toLowerCase().includes(e.name.toLowerCase()));

      if (exercise) {
        setActiveExercise(exercise);
        setFeedback(`Detected: ${exercise.name}. Ready!`);
        // Reset state for new exercise
        const now = performance.now();
        repState.current = { 
          phase: 'up', 
          count: 0, 
          calibratedMin: 180, 
          calibratedMax: 0, 
          lastAngle: null, 
          currentRepRom: 0, 
          phaseStartTime: now, 
          lastEccentricTime: 0, 
          history: [], 
          lastMinTime: now, 
          lastMaxTime: now, 
          lastMoveTime: now,
          lastFrameTime: now,
          preferredSide: null
        };
      } else {
        setFeedback("Couldn't identify. Try again.");
      }
    } catch (err) {
      console.error("Detection error:", err);
      setFeedback("AI Scan failed.");
    } finally {
      setIsDetecting(false);
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: 640, height: 480 } });
      if (videoRef.current) { 
        videoRef.current.srcObject = stream; 
        videoRef.current.onloadedmetadata = () => { 
          videoRef.current?.play(); 
          setCameraActive(true);
          cameraActiveRef.current = true;
          processVideo();
        }; 
      }
    } catch (err) { 
      setError("Camera access denied."); 
    }
  };

  const processVideo = async () => {
    if (videoRef.current && cameraActiveRef.current && poseRef.current) {
      try {
        if (videoRef.current.readyState >= 2 && videoRef.current.videoWidth > 0) {
          await poseRef.current.send({ image: videoRef.current });
        }
      } catch (err) {
        console.error("Pose send error:", err);
      }
      requestRef.current = requestAnimationFrame(processVideo);
    }
  };

  const stopCameraAndAnalyze = () => {
    if (videoRef.current?.srcObject) {
      (videoRef.current.srcObject as MediaStream).getTracks().forEach(t => t.stop());
    }
    setCameraActive(false); 
    cameraActiveRef.current = false;
    if (requestRef.current) cancelAnimationFrame(requestRef.current);
    const h = repState.current.history;
    if (h.length > 0) {
      const total = h.length; 
      const avgRom = Math.round(h.reduce((s: any, r: any) => s + r.rom, 0) / total); 
      const avgEcc = h.reduce((s: any, r: any) => s + r.eccentricTime, 0) / total;
      let feed = []; 
      let score = 100;

      // 1. Extreme ROM Penalty
      if (avgRom < 99) { 
        const romDeficit = 99 - avgRom;
        score -= (romDeficit * 10); // 5% deficit = -50 points
        feed.push(`ROM DEFICIT: ${avgRom}% range. You are cutting your reps short. Hit the full stretch and peak contraction.`); 
      }

      // 2. Extreme Tempo Penalty (Eccentric/Negative phase)
      // Ideal is 2.5s - 3s.
      if (avgEcc < 2500) { 
        const tempoDeficit = (2500 - avgEcc) / 100;
        score -= (tempoDeficit * 6); // 1s too fast (1000ms deficit) = -60 points
        feed.push(`TEMPO ERROR: Your negative was only ${(avgEcc/1000).toFixed(1)}s. You're losing 50% of the gains by dropping the weight. Aim for 2.5s+`); 
      }

      // 3. Rep Count & Weight Guidance
      if (currentSet !== 0) { // Only for working sets
        if (total < 5) {
          score -= 15;
          feed.push("TOO HEAVY: You failed to hit 5 reps. Lower the weight by 10-15% to stay in a productive growth range.");
        } else if (total > 14) {
          feed.push("TOO LIGHT: 15+ reps detected. Increase the weight next set to stay in the 8-12 hypertrophy range.");
        }
      }

      // 4. Consistency Penalty
      if (total > 2) {
        const roms = h.map((r: any) => r.rom);
        const romVariance = roms.reduce((a: number, b: number) => a + Math.pow(b - avgRom, 2), 0) / total;
        if (romVariance > 15) { // Stricter variance (was 25)
          score -= 20;
          feed.push(`MOPPY REPS: Your range of motion is inconsistent. Every rep must be a carbon copy of the last.`);
        }
      }

      // 5. Elite Form Requirement
      if (score >= 95 && avgRom >= 99 && avgEcc >= 2800) {
        feed.push("ELITE EXECUTION: Perfect control, depth, and consistency. This is how champions train.");
      } else if (score > 80) {
        feed.push("SOLID WORK: Good form, but there's still room for more control on the negative.");
      }

      const newStat = { 
        id: Date.now(),
        date: new Date().toISOString(),
        exercise: activeExercise.name, 
        reps: total, 
        avgRom, 
        timeUnderTension: Math.round(h.reduce((s: any, r: any) => s + r.eccentricTime + r.concentricTime, 0) / 1000), 
        score: Math.max(0, Math.min(100, Math.round(score))), 
        feedback: feed, 
        isWarmup: currentSet === 0, 
        currentSetNum: currentSet, 
        totalSets: targetSets 
      };

      setWorkoutStats(newStat);
      setWorkoutHistory(prev => [newStat, ...prev].slice(0, 50)); // Keep last 50 sets
      setView('summary');
    } else handleTransitionFromSummary(true);
  };

  const handleTransitionFromSummary = (skipped = false) => {
    if ((currentSet >= targetSets && currentSet !== 0) || skipped) {
      const idx = routines[selectedDay].indexOf(activeExercise.id);
      const nextId = routines[selectedDay][idx + 1]; if (nextId) switchExercise(nextId); else setView('schedule');
    } else { setRestTimeLeft(currentSet === 0 ? 60 : (activeExercise.isCompound ? 120 : 90)); setView('rest'); }
  };

  const finishRest = () => {
    setCurrentSet(v => v + 1); setReps(0);
    const now = performance.now();
    repState.current = { ...repState.current, phase: 'up', count: 0, calibratedMin: 180, calibratedMax: 0, currentRepRom: 0, phaseStartTime: now, history: [], lastMoveTime: now };
    setView('workout'); startCamera();
  };

  const toggleExerciseInRoutine = (id: string) => {
    setRoutines((prev: any) => {
      const day = prev[selectedDay];
      return { ...prev, [selectedDay]: day.includes(id) ? day.filter((ex: string) => ex !== id) : [...day, id] };
    });
  };

  const handleDragStart = (e: React.DragEvent, index: number) => { 
    setDraggedIndex(index); 
    e.dataTransfer.setData('text/plain', index.toString()); 
  };
  const handleDragOver = (e: React.DragEvent) => e.preventDefault();
  const handleDrop = (e: React.DragEvent, dropIndex: number) => {
    e.preventDefault(); if (draggedIndex === null || draggedIndex === dropIndex) return;
    setRoutines((prev: any) => {
      const curr = [...prev[selectedDay]]; const item = curr[draggedIndex];
      curr.splice(draggedIndex, 1); curr.splice(dropIndex, 0, item);
      return { ...prev, [selectedDay]: curr };
    });
    setDraggedIndex(null);
  };
  const handleDragEnd = () => setDraggedIndex(null);

  const drawSkeleton = (landmarks: any[], ctx: CanvasRenderingContext2D) => {
    if (!landmarks || !Array.isArray(landmarks)) return;
    const width = canvasRef.current?.width || 640;
    const height = canvasRef.current?.height || 480;

    landmarks.forEach(p => { 
      if (p && p.visibility > 0.2) { 
        ctx.beginPath(); 
        ctx.arc(p.x * width, p.y * height, 4, 0, 2*Math.PI); 
        ctx.fillStyle = '#C4B5FD'; 
        ctx.fill(); 
      } 
    });

    // Simple skeleton connections for MediaPipe
    const connections = [
      [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], // Upper
      [11, 23], [12, 24], [23, 24], // Torso
      [23, 25], [25, 27], [24, 26], [26, 28] // Lower
    ];

    connections.forEach(([i, j]) => {
      const p1 = landmarks[i];
      const p2 = landmarks[j];
      if (p1 && p2 && p1.visibility > 0.2 && p2.visibility > 0.2) {
        ctx.beginPath();
        ctx.moveTo(p1.x * width, p1.y * height);
        ctx.lineTo(p2.x * width, p2.y * height);
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#8B5CF6';
        ctx.stroke();
      }
    });
  };

  const feedbackColorClass = feedback.includes('Perfect') || feedback.includes('Great')
    ? 'bg-emerald-900/50 border-emerald-500/30 text-emerald-300' 
    : (feedback.includes('fast') || feedback.includes('body') || feedback.includes('frame') || feedback.includes('next time')) 
      ? 'bg-rose-900/50 border-rose-500/30 text-rose-300' 
      : 'bg-indigo-900/50 border-indigo-500/30 text-indigo-100';

  if (isAuthLoading) return <div className="h-[100dvh] w-full bg-[#0a0a1a] flex items-center justify-center"><RefreshCw className="animate-spin text-indigo-500" /></div>;

  if (!activeAccount) return (
    <div className="h-[100dvh] w-full bg-[#0a0a1a] text-white flex flex-col items-center justify-center p-6 font-sans">
      <div className="w-full max-w-sm animate-in fade-in zoom-in-95 duration-500">
        <div className="w-16 h-16 bg-indigo-600 rounded-[1.5rem] flex items-center justify-center mb-8 mx-auto shadow-2xl relative">
          <Activity size={32} />
          <div className="absolute inset-0 bg-indigo-600 rounded-[1.5rem] blur-xl opacity-40 animate-pulse" />
        </div>
        <h1 className="text-3xl font-black text-center mb-2 tracking-tight uppercase italic">FormPerfect</h1>
        <p className="text-indigo-400/60 text-center mb-10 text-xs font-black uppercase tracking-[0.2em]">{authMode === 'login' ? 'Continue Training' : 'Register Account'}</p>
        
        {error && <div className="bg-rose-900/40 border border-rose-500/30 p-4 rounded-2xl mb-6 text-rose-400 text-sm font-bold text-center animate-in slide-in-from-top-2">{error}</div>}
        {authSuccess && <div className="bg-emerald-900/40 border border-emerald-500/30 p-4 rounded-2xl mb-6 text-emerald-400 text-sm font-bold text-center animate-in slide-in-from-top-2">{authSuccess}</div>}

        <form onSubmit={handleAuth} className="space-y-4">
          <input placeholder="Username" value={username} onChange={(e) => setUsername(e.target.value)} className="w-full bg-indigo-900/20 border border-indigo-500/20 rounded-2xl p-4 text-white focus:outline-none focus:border-indigo-500 transition-colors" required />
          <input type="password" placeholder="Password" value={password} onChange={(e) => setPassword(e.target.value)} className="w-full bg-indigo-900/20 border border-indigo-500/20 rounded-2xl p-4 text-white focus:outline-none focus:border-indigo-500 transition-colors" required />
          {authError && <p className="text-rose-400 text-xs text-center font-bold">{authError}</p>}
          <button type="submit" className="w-full bg-indigo-600 hover:bg-indigo-500 text-white p-4 rounded-2xl font-black shadow-xl transition-all active:scale-95 uppercase tracking-widest text-sm">
            {authMode === 'login' ? 'Enter Gym' : 'Sign Up'}
          </button>
        </form>
        <button onClick={() => { setAuthMode(authMode === 'login' ? 'signup' : 'login'); setAuthError(''); setAuthSuccess(''); }} className="w-full text-indigo-400/60 mt-8 text-[10px] font-black hover:text-indigo-400 transition-colors uppercase tracking-widest text-center block">
          {authMode === 'login' ? "New here? Create Account" : "Back to login"}
        </button>
      </div>
    </div>
  );

  return (
    <div className="h-[100dvh] w-full bg-[#0a0a1a] text-slate-100 font-sans flex flex-col items-center overflow-hidden">
      <div className="fixed top-[-20%] right-[-30%] w-[100vw] h-[100vw] bg-indigo-600/10 blur-[150px] rounded-full pointer-events-none" />
      <div className="fixed bottom-[-20%] left-[-30%] w-[100vw] h-[100vw] bg-violet-600/10 blur-[150px] rounded-full pointer-events-none" />

      <div className="w-full max-w-md mx-auto h-full flex flex-col relative p-4 overflow-hidden">
        
        {isProfileOpen && (
          <div className="fixed inset-0 bg-black/80 backdrop-blur-md z-[60] animate-in fade-in flex items-center justify-center p-6">
            <div className="bg-indigo-950 w-full max-w-sm rounded-[2.5rem] p-8 border border-indigo-500/20 shadow-2xl relative">
              <button onClick={() => setIsProfileOpen(false)} className="absolute top-6 right-6 text-indigo-400 hover:text-white transition-colors"><XCircle /></button>
              <h2 className="text-2xl font-black mb-6 uppercase tracking-tight italic">Profile</h2>
              <div className="flex justify-center gap-3 mb-8">
                {AVATARS.map(a => (
                  <button key={a} onClick={() => setProfile({...profile, pfp: a})} className={`w-12 h-12 rounded-[1.25rem] overflow-hidden border-2 transition-all ${profile.pfp === a ? 'border-violet-500 scale-110 shadow-lg shadow-violet-500/20' : 'border-transparent opacity-40'}`}>
                    <img src={a} alt="avatar" />
                  </button>
                ))}
              </div>
              <form onSubmit={handleUpdateProfile} className="space-y-4">
                <input value={profile.name} onChange={(e) => setProfile({...profile, name: e.target.value})} className="w-full bg-indigo-900/40 border border-indigo-500/10 rounded-2xl p-4 focus:border-indigo-500 outline-none" placeholder="Name" />
                <textarea value={profile.bio} onChange={(e) => setProfile({...profile, bio: e.target.value})} className="w-full bg-indigo-900/40 border border-indigo-500/10 rounded-2xl p-4 focus:border-indigo-500 outline-none h-24 resize-none" placeholder="Bio" />
                <button type="submit" className="w-full bg-indigo-600 p-4 rounded-2xl font-black shadow-lg uppercase tracking-widest text-sm hover:bg-indigo-500 transition-colors">SAVE CHANGES</button>
                <button type="button" onClick={handleLogout} className="w-full bg-rose-500/10 text-rose-400 p-4 rounded-2xl font-bold flex items-center justify-center gap-2 uppercase tracking-widest text-sm active:bg-rose-500 active:text-white transition-all"><LogOut size={16}/> LOGOUT</button>
              </form>
            </div>
          </div>
        )}

        {view === 'schedule' && (
          <div className="flex flex-col h-full w-full animate-in fade-in pb-20">
            <div className="flex justify-between items-center mb-6 pt-4 px-2 shrink-0">
              <div className="overflow-hidden">
                <h2 className="text-3xl font-black text-white tracking-tight leading-none truncate max-w-[200px]">Hi, {profile.name}</h2>
                <p className="text-indigo-300/50 text-[10px] font-bold mt-1 uppercase tracking-widest truncate max-w-[200px]">{profile.bio}</p>
              </div>
              <button onClick={() => setIsProfileOpen(true)} className="w-12 h-12 rounded-2xl overflow-hidden border border-indigo-500/20 bg-indigo-900/40 flex items-center justify-center shadow-xl relative group flex-shrink-0">
                <img src={profile.pfp} alt="user" className="w-full h-full object-cover" />
                <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity"><Edit2 size={16}/></div>
              </button>
            </div>
            <div className="flex overflow-x-auto hide-scrollbar gap-2 mb-6 px-2 shrink-0">
              {DAYS.map(day => (
                <button key={day} onClick={() => setSelectedDay(day)} className={`px-5 py-3 rounded-2xl font-bold transition-all shrink-0 text-sm ${selectedDay === day ? 'bg-indigo-600 text-white shadow-lg' : 'bg-indigo-950/40 text-indigo-400 border border-indigo-900/30'}`}>
                  {day}
                </button>
              ))}
            </div>
            <div className="flex-1 overflow-y-auto min-h-0 hide-scrollbar pb-32 px-2">
              <div className="bg-indigo-950/20 rounded-[2rem] p-5 border border-indigo-800/40 shadow-xl">
                <div className="flex justify-between items-center mb-4 px-1">
                  <h2 className="text-lg font-bold text-white uppercase tracking-wider">{selectedDay} Routine</h2>
                  <span className="text-indigo-400/60 text-[9px] font-black uppercase tracking-widest">{routines[selectedDay]?.length || 0} Items</span>
                </div>
                <div className="space-y-3">
                  {(routines[selectedDay] || []).map((id: string, idx: number) => {
                    const ex = EXERCISE_DB.find(e => e.id === id);
                    if (!ex) return null;
                    return (
                      <div key={`${id}-${idx}`} draggable onDragStart={(e) => handleDragStart(e, idx)} onDragOver={handleDragOver} onDrop={(e) => handleDrop(e, idx)} onDragEnd={handleDragEnd}
                        className={`flex items-center justify-between p-3 rounded-2xl bg-indigo-900/30 border transition-all ${draggedIndex === idx ? 'opacity-30 border-indigo-50 scale-95' : 'border-indigo-500/10'}`}
                      >
                        <div className="flex items-center gap-3">
                          <GripVertical className="w-5 h-5 text-indigo-800 cursor-grab" />
                          <span className="text-violet-500 font-black text-sm">{idx + 1}.</span>
                          <div><p className="font-bold text-white leading-tight text-sm">{ex.name}</p><p className="text-[9px] text-indigo-400/50 uppercase tracking-wider">{ex.muscleGroup}</p></div>
                        </div>
                        <div className="flex gap-1">
                          <button onClick={() => switchExercise(id)} className="w-10 h-10 flex items-center justify-center bg-indigo-600 text-white rounded-xl active:scale-90 transition-all shadow-lg shadow-indigo-600/20"><Play size={16} fill="currentColor" /></button>
                          <button onClick={() => toggleExerciseInRoutine(id)} className="w-10 h-10 flex items-center justify-center bg-indigo-950/50 text-indigo-700 rounded-xl active:scale-90 transition-all"><Trash2 size={16} /></button>
                        </div>
                      </div>
                    );
                  })}
                  {(!routines[selectedDay] || routines[selectedDay].length === 0) && <div className="text-center py-12 opacity-30 flex flex-col items-center"><Dumbbell className="w-10 h-10 mb-2 text-indigo-400"/><p className="text-xs font-bold text-indigo-300">Rest Day</p></div>}
                </div>
              </div>
            </div>
            <div className="fixed bottom-0 left-0 right-0 p-4 flex justify-center max-w-md mx-auto w-full z-20">
              <div className="flex bg-indigo-900/90 backdrop-blur-2xl p-2 rounded-[2rem] shadow-2xl border border-indigo-500/20 gap-2 w-full">
                <button onClick={() => setView('schedule')} className={`flex-1 flex flex-col items-center py-3 rounded-2xl ${view === 'schedule' ? 'text-violet-400 bg-violet-500/10' : 'text-indigo-400/60'}`}><Calendar className="w-5 h-5"/></button>
                <button onClick={() => setView('library')} className={`flex-1 flex flex-col items-center py-3 rounded-2xl text-indigo-400/60`}><Plus className="w-5 h-5"/></button>
                <button onClick={() => setView('history')} className={`flex-1 flex flex-col items-center py-3 rounded-2xl ${view === 'history' ? 'text-violet-400 bg-violet-500/10' : 'text-indigo-400/60'}`}><BarChart2 className="w-5 h-5"/></button>
                <button onClick={handleSaveAll} className={`flex-1 flex flex-col items-center py-3 rounded-2xl transition-colors ${showSaved ? 'text-emerald-400' : 'text-indigo-400/60'}`}><Save className="w-5 h-5"/></button>
                <button onClick={startWorkout} className="flex-[2.5] bg-indigo-600 text-white p-3 rounded-2xl font-black text-sm flex items-center justify-center gap-2 active:scale-95 transition-transform"><Play className="w-4 h-4 fill-current"/> START</button>
              </div>
            </div>
          </div>
        )}

        {view === 'history' && (
          <div className="flex flex-col h-full w-full animate-in fade-in pb-20 px-1">
            <div className="flex items-center justify-between mb-6 sticky top-0 bg-indigo-950/90 backdrop-blur-xl py-4 z-10 px-2">
              <h2 className="text-xl font-black text-white uppercase tracking-tight italic">History</h2>
              <button onClick={() => setView('schedule')} className="p-2.5 bg-indigo-900 rounded-full border border-indigo-700 shadow-xl"><ArrowLeft size={20} className="text-white"/></button>
            </div>
            <div className="flex-1 overflow-y-auto hide-scrollbar space-y-4 pb-20 px-2">
              {workoutHistory.length === 0 ? (
                <div className="text-center py-20 opacity-30 flex flex-col items-center">
                  <BarChart2 className="w-12 h-12 mb-4 text-indigo-400"/>
                  <p className="text-sm font-bold text-indigo-300">No workouts recorded yet.</p>
                </div>
              ) : (
                workoutHistory.map((stat) => (
                  <div key={stat.id} className="bg-indigo-900/30 border border-indigo-500/10 p-4 rounded-[1.5rem] space-y-3">
                    <div className="flex justify-between items-start">
                      <div>
                        <h4 className="font-black text-white text-sm uppercase tracking-tight">{stat.exercise}</h4>
                        <p className="text-[10px] text-indigo-400 font-bold">{new Date(stat.date).toLocaleDateString()} • {new Date(stat.date).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</p>
                      </div>
                      <div className="bg-indigo-600 px-2 py-1 rounded-lg text-[10px] font-black text-white">SCORE: {stat.score}</div>
                    </div>
                    <div className="grid grid-cols-3 gap-2">
                      <div className="bg-black/20 p-2 rounded-xl text-center">
                        <p className="text-[8px] text-indigo-400 uppercase font-black">Reps</p>
                        <p className="text-sm font-black text-white">{stat.reps}</p>
                      </div>
                      <div className="bg-black/20 p-2 rounded-xl text-center">
                        <p className="text-[8px] text-indigo-400 uppercase font-black">ROM</p>
                        <p className="text-sm font-black text-white">{stat.avgRom}%</p>
                      </div>
                      <div className="bg-black/20 p-2 rounded-xl text-center">
                        <p className="text-[8px] text-indigo-400 uppercase font-black">TUT</p>
                        <p className="text-sm font-black text-white">{stat.timeUnderTension}s</p>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
            <div className="fixed bottom-0 left-0 right-0 p-4 flex justify-center max-w-md mx-auto w-full z-20">
              <div className="flex bg-indigo-900/90 backdrop-blur-2xl p-2 rounded-[2rem] shadow-2xl border border-indigo-500/20 gap-2 w-full">
                <button onClick={() => setView('schedule')} className={`flex-1 flex flex-col items-center py-3 rounded-2xl ${view === 'schedule' ? 'text-violet-400 bg-violet-500/10' : 'text-indigo-400/60'}`}><Calendar className="w-5 h-5"/></button>
                <button onClick={() => setView('library')} className={`flex-1 flex flex-col items-center py-3 rounded-2xl text-indigo-400/60`}><Plus className="w-5 h-5"/></button>
                <button onClick={() => setView('history')} className={`flex-1 flex flex-col items-center py-3 rounded-2xl ${view === 'history' ? 'text-violet-400 bg-violet-500/10' : 'text-indigo-400/60'}`}><BarChart2 className="w-5 h-5"/></button>
                <button onClick={handleSaveAll} className={`flex-1 flex flex-col items-center py-3 rounded-2xl transition-colors ${showSaved ? 'text-emerald-400' : 'text-indigo-400/60'}`}><Save className="w-5 h-5"/></button>
                <button onClick={startWorkout} className="flex-[2.5] bg-indigo-600 text-white p-3 rounded-2xl font-black text-sm flex items-center justify-center gap-2 active:scale-95 transition-transform"><Play className="w-4 h-4 fill-current"/> START</button>
              </div>
            </div>
          </div>
        )}

        {view === 'library' && (
          <div className="flex flex-col h-full w-full animate-in fade-in pb-6 px-1">
            <div className="flex items-center justify-between mb-6 sticky top-0 bg-indigo-950/90 backdrop-blur-xl py-4 z-10 px-2">
              <h2 className="text-xl font-black text-white uppercase tracking-tight italic">Library</h2>
              <button onClick={() => setView('schedule')} className="p-2.5 bg-indigo-900 rounded-full border border-indigo-700 shadow-xl"><ArrowLeft size={20} className="text-white"/></button>
            </div>
            <div className="flex-1 overflow-y-auto hide-scrollbar space-y-6 pb-20 px-2">
              {MUSCLE_GROUPS.map(muscle => (
                <div key={muscle} className="space-y-2">
                  <h4 className="text-[10px] font-black text-violet-400 uppercase tracking-widest ml-2">{muscle}</h4>
                  <div className="grid grid-cols-1 gap-2">
                    {EXERCISE_DB.filter(e => e.muscleGroup === muscle).map(ex => {
                      const isAdded = routines[selectedDay]?.includes(ex.id);
                      return (
                        <button key={ex.id} onClick={() => toggleExerciseInRoutine(ex.id)} className={`w-full flex items-center justify-between p-3.5 rounded-2xl border transition-all ${isAdded ? 'bg-indigo-600 border-indigo-400 text-white' : 'bg-indigo-900/30 border-indigo-500/10 text-indigo-200'}`}>
                          <span className="font-bold text-sm">{ex.name}</span>
                          {isAdded ? <CheckCircle size={20} /> : <Plus size={20} className="opacity-40" />}
                        </button>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {view === 'workout' && (
          <div className="flex flex-col h-full w-full animate-in zoom-in-95 duration-300">
            <div className="flex items-center justify-between mb-3 shrink-0 pt-2 px-2">
              <button onClick={() => setView('schedule')} className="p-2.5 bg-indigo-900 rounded-full"><XCircle size={20}/></button>
              <div className="flex gap-1 overflow-x-auto hide-scrollbar max-w-[65%] px-2">
                {routines[selectedDay].map((id: string, idx: number) => (
                   <button key={id} onClick={() => switchExercise(id)} className={`min-w-[32px] h-8 rounded-full flex items-center justify-center text-[10px] font-black ${activeExercise?.id === id ? 'bg-violet-500 text-white shadow-lg' : 'bg-indigo-950 text-indigo-400'}`}>{idx + 1}</button>
                ))}
              </div>
            </div>
            <div className="flex-1 relative bg-indigo-900 rounded-[2.5rem] overflow-hidden border border-indigo-500/20 shadow-2xl flex flex-col mb-3">
              {!isSetupPhase && <div className="absolute top-4 left-4 z-20 bg-indigo-950/80 backdrop-blur-xl px-3 py-1.5 rounded-full border border-indigo-500/20 shadow-xl flex items-center gap-2"><p className="text-[9px] font-black text-white uppercase tracking-widest">{activeExercise?.name}</p><span className="w-1.5 h-1.5 bg-violet-500 rounded-full animate-pulse shadow-[0_0_5px_#8B5CF6]" /></div>}
              <video ref={videoRef} className={`absolute inset-0 w-full h-full object-cover transform scale-x-[-1] ${!cameraActive ? 'hidden' : ''}`} playsInline muted />
              <canvas ref={canvasRef} width={640} height={480} className={`absolute inset-0 w-full h-full object-cover transform scale-x-[-1] z-10 ${!cameraActive ? 'hidden' : ''}`} />
              {cameraActive && (
                <div className="absolute top-4 right-4 z-30">
                  <button 
                    onClick={detectExercise} 
                    disabled={isDetecting}
                    className={`p-3 rounded-2xl backdrop-blur-xl border border-white/20 shadow-2xl transition-all active:scale-95 flex items-center gap-2 ${isDetecting ? 'bg-violet-500/50 text-white animate-pulse' : 'bg-indigo-950/80 text-indigo-300 hover:text-white'}`}
                  >
                    <Zap size={18} className={isDetecting ? 'animate-spin' : ''} />
                    <span className="text-[10px] font-black uppercase tracking-widest">{isDetecting ? 'Scanning...' : 'Auto-Detect'}</span>
                  </button>
                </div>
              )}
              {cameraActive && (
                <div className="absolute bottom-0 left-0 w-full z-20 bg-gradient-to-t from-black/80 to-transparent p-6 pt-16">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-[10px] font-black text-indigo-300 uppercase tracking-widest">Form Quality</span>
                    <span className={`text-[10px] font-black px-2 py-0.5 rounded-full ${
                      formScore > 90 ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' :
                      formScore > 70 ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30' :
                      'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                    }`}>
                      {formScore > 90 ? 'EXCELLENT' : formScore > 70 ? 'GOOD' : 'IMPROVE'}
                    </span>
                  </div>
                  <div className="flex justify-between items-end mb-2">
                    <div ref={progressLabelRef} className="text-[9px] font-black text-indigo-300 tracking-widest uppercase">Analyzing</div>
                    <div ref={progressTextRef} className="text-5xl font-black text-white tabular-nums drop-shadow-2xl leading-none">0%</div>
                  </div>
                  <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden shadow-inner border border-white/5">
                    <div ref={progressFillRef} className="h-full w-0 bg-violet-500 rounded-full transition-all duration-75 shadow-[0_0_10px_#8B5CF6]" />
                  </div>
                  <div className="flex justify-between mt-3">
                    <div className="flex items-center gap-2">
                      <div className={`w-1.5 h-1.5 rounded-full ${formScore > 85 ? 'bg-emerald-500 animate-pulse' : 'bg-white/20'}`} />
                      <span className="text-[8px] text-white/40 font-black uppercase tracking-widest">Stability</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className={`w-1.5 h-1.5 rounded-full ${formScore > 70 ? 'bg-cyan-500 animate-pulse' : 'bg-white/20'}`} />
                      <span className="text-[8px] text-white/40 font-black uppercase tracking-widest">Tempo</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className={`w-1.5 h-1.5 rounded-full ${formScore > 50 ? 'bg-indigo-500 animate-pulse' : 'bg-white/20'}`} />
                      <span className="text-[8px] text-white/40 font-black uppercase tracking-widest">ROM</span>
                    </div>
                  </div>
                </div>
              )}
              {!cameraActive && (
                <div className="absolute inset-0 flex flex-col items-center justify-center p-8 bg-[#0a0a1a] z-20 text-center">
                  {isSetupPhase ? (
                    <div className="w-full max-w-sm animate-in fade-in scale-in-95 duration-300">
                      <h3 className="text-xl font-black text-white mb-6 uppercase tracking-tight leading-tight">{activeExercise?.name} Setup</h3>
                      <div className="bg-indigo-900/40 p-5 rounded-3xl space-y-4 mb-8 border border-indigo-500/10 shadow-xl">
                         <div className="flex justify-between items-center"><span className="text-sm font-bold text-indigo-200">Sets</span><div className="flex gap-4 items-center"><button onClick={() => setTargetSets(Math.max(1, targetSets-1))} className="p-1.5 bg-indigo-800 rounded-lg text-white active:scale-90"><Minus size={16}/></button><span className="text-xl font-black text-white">{targetSets}</span><button onClick={() => setTargetSets(targetSets+1)} className="p-1.5 bg-indigo-800 rounded-lg text-white active:scale-90"><Plus size={16}/></button></div></div>
                         <div className="flex justify-between items-center"><span className="text-sm font-bold uppercase">Warm-up Set</span><button onClick={() => setIncludeWarmup(!includeWarmup)} className={`w-11 h-6 rounded-full relative transition-colors ${includeWarmup ? 'bg-violet-600' : 'bg-indigo-800'}`}><div className={`w-3 h-3 bg-white rounded-full absolute top-1 transition-all shadow-sm ${includeWarmup ? 'left-6' : 'left-1'}`} /></button></div>
                      </div>
                      <button onClick={handleStartSetPhase} className="w-full bg-indigo-600 p-4 rounded-2xl font-black shadow-xl shadow-indigo-600/20 text-white uppercase tracking-widest text-sm active:scale-95 transition-all">Begin Set</button>
                    </div>
                  ) : (
                    <div className="w-full animate-in fade-in scale-in-95 duration-300">
                      <div className="w-16 h-16 bg-indigo-900 rounded-2xl flex items-center justify-center mx-auto mb-6 border border-indigo-700 text-violet-400 shadow-2xl relative">
                        <Activity size={32} />
                        <div className="absolute inset-0 bg-violet-400 opacity-20 blur-lg animate-pulse" />
                      </div>
                      <h3 className="text-lg font-black text-white mb-2 uppercase tracking-tighter italic">{currentSet === 0 ? 'Warm-up Set' : `Set ${currentSet} ready`}</h3>
                      <p className="text-xs text-indigo-400/60 mb-8 max-w-[200px] mx-auto leading-relaxed uppercase tracking-widest font-black">AI detects end of set automatically</p>
                      <button onClick={startCamera} className="w-full bg-indigo-600 p-4 rounded-2xl font-black text-white active:scale-95 uppercase tracking-widest text-sm shadow-xl hover:bg-indigo-50 transition-all">Start Camera</button>
                    </div>
                  )}
                </div>
              )}
            </div>
            <div className={`p-4 rounded-2xl border mb-3 shadow-lg text-sm font-bold flex gap-3 items-center transition-all backdrop-blur-md ${feedbackColorClass} shrink-0`}>
              <Info size={20} className="shrink-0 opacity-50" /><p className="leading-tight">{feedback}</p>
            </div>
            <div className="grid grid-cols-2 gap-4 mb-2 shrink-0 px-1">
              <div className="bg-indigo-900/40 rounded-[2rem] p-4 relative border border-indigo-500/10 flex flex-col items-center justify-center shadow-xl">
                <button onClick={resetReps} className="absolute top-3 right-3 text-indigo-800 hover:text-indigo-400 transition-colors"><RotateCcw size={16} /></button>
                <span className="text-[10px] font-black text-indigo-500 uppercase tracking-widest mb-1">Live Reps</span>
                <span className="text-5xl font-black text-white tabular-nums leading-none">{reps}</span>
              </div>
              <button onClick={stopCameraAndAnalyze} disabled={!cameraActive} className="bg-violet-600 text-white rounded-[2rem] p-4 font-black flex flex-col items-center justify-center gap-1 uppercase tracking-widest text-[10px] active:scale-95 disabled:opacity-30 transition-all shadow-xl shadow-violet-600/20">
                <CheckCircle size={28} /> Manual Stop
              </button>
            </div>
          </div>
        )}

        {view === 'summary' && (
          <div className="flex flex-col h-full w-full animate-in fade-in pb-4 px-1">
            <div className="text-center mb-6 pt-4 shrink-0">
              <div className={`w-14 h-14 rounded-[1.25rem] mx-auto mb-3 flex items-center justify-center border-2 ${workoutStats.score > 80 ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30 shadow-[0_0_15px_rgba(16,185,129,0.3)]' : 'bg-rose-500/20 text-rose-400 border-rose-500/30'}`}>
                 <Target size={32} />
              </div>
              <h2 className="text-2xl font-black text-white tracking-tight leading-none italic uppercase">Set Analysis</h2>
              <p className="text-indigo-400 text-[9px] font-black uppercase tracking-widest mt-2">{workoutStats.exercise}</p>
            </div>
            <div className="grid grid-cols-4 gap-2 mb-6 shrink-0 px-1">
              {[ {l: 'REPS', v: workoutStats.reps, i: <Activity size={16}/>, c: 'text-indigo-400' }, {l: 'ROM', v: workoutStats.avgRom+'%', i: <BarChart2 size={16}/>, c: 'text-violet-400' }, {l: 'SEC', v: workoutStats.timeUnderTension, i: <Clock size={16}/>, c: 'text-blue-400' }, {l: 'SCORE', v: workoutStats.score, i: <Zap size={16}/>, c: 'text-amber-400' } ].map(s => (
                <div key={s.l} className="bg-indigo-900/30 py-4 px-1 rounded-2xl border border-indigo-500/10 text-center shadow-lg">
                  <div className={`mb-1 flex justify-center ${s.c}`}>{s.i}</div>
                  <div className="text-lg font-black text-white leading-none mb-1 tabular-nums">{s.v}</div>
                  <div className="text-[8px] font-black text-indigo-500 uppercase tracking-widest leading-none">{s.l}</div>
                </div>
              ))}
            </div>
            <div className="flex-1 overflow-y-auto hide-scrollbar bg-indigo-950/40 rounded-[2.5rem] p-6 mb-6 border border-indigo-800 shadow-xl min-h-0">
              <h3 className="text-[10px] font-black text-indigo-500 uppercase tracking-widest mb-4 border-b border-indigo-800 pb-3">AI Form Feedback</h3>
              <div className="space-y-4">
                {workoutStats.feedback.map((f: string, i: number) => (
                  <div key={i} className="flex gap-4 items-start"><div className="w-1.5 h-1.5 rounded-full bg-rose-500 mt-1.5 shrink-0 shadow-[0_0_10px_#ef4444]" /><p className="text-[15px] font-bold text-indigo-100 leading-snug">{f}</p></div>
                ))}
                {workoutStats.feedback.length === 0 && <p className="text-xs font-bold text-emerald-400 italic">Excellent set. No form errors detected.</p>}
              </div>
            </div>
            <button onClick={() => handleTransitionFromSummary()} className="w-full bg-indigo-600 text-white p-5 rounded-2xl font-black flex justify-center items-center gap-3 active:scale-95 shadow-xl uppercase tracking-widest text-sm hover:bg-indigo-500 transition-all">
              CONTINUE <ArrowRight size={20} />
            </button>
          </div>
        )}

        {view === 'rest' && (
          <div className="flex flex-col items-center justify-center h-full w-full animate-in fade-in pb-10 px-8 text-center shrink-0">
            <Timer size={48} className="text-violet-400 mb-6 drop-shadow-2xl animate-pulse" />
            <h2 className="text-2xl font-black text-white mb-1 uppercase tracking-tight italic">Recovery Phase</h2>
            <p className="text-indigo-400 text-[10px] font-bold tracking-[0.25em] mb-12 uppercase">Preparing Set {currentSet+1}</p>
            <div className="text-[10rem] font-black text-white mb-14 tabular-nums tracking-tighter leading-none drop-shadow-[0_20px_40px_rgba(99,102,241,0.3)]">
              {Math.floor(restTimeLeft/60)}:{(restTimeLeft%60).toString().padStart(2, '0')}
            </div>
            <div className="flex gap-5 w-full max-w-sm mb-12 shrink-0">
              <button onClick={() => setRestTimeLeft(t => Math.max(0, t-30))} className="flex-1 py-5 bg-indigo-900/40 border border-indigo-500/10 shadow-xl rounded-[1.5rem] font-black text-indigo-300 text-lg active:scale-90 transition-transform">-30s</button>
              <button onClick={() => setRestTimeLeft(t => t+30)} className="flex-1 py-5 bg-indigo-900/40 border border-indigo-500/10 shadow-xl rounded-[1.5rem] font-black text-indigo-300 text-lg active:scale-90 transition-transform">+30s</button>
            </div>
            <button onClick={finishRest} className="w-full max-w-sm bg-indigo-600 text-white p-6 rounded-[2rem] font-black text-xl shadow-2xl active:scale-95 uppercase tracking-widest hover:bg-indigo-500 transition-all">Resume Now</button>
          </div>
        )}

      </div>
    </div>
  );
}
