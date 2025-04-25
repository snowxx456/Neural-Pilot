// components/TrainingProgress.tsx
import React, { useEffect, useState } from 'react';
import styles from './TrainingProgress.module.css'; // Adjust the path as necessary

interface StepDetails {
  [key: string]: any;
}

interface TrainingStep {
  id: number;
  name: string;
  status: 'pending' | 'processing' | 'completed' | 'error';
  details: StepDetails;
}

const TrainingProgress: React.FC = () => {
  const [steps, setSteps] = useState<{ [key: number]: TrainingStep }>({});
  const [connected, setConnected] = useState(false);
  const [log, setLog] = useState<string[]>([]);
  const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/';
  const [datasetId,setDatasetId] = useState<any | null>(null); // Assuming datasetId is set somewhere in your app
  const [datasetName,setDatasetName] = useState<any | null>(null); // Assuming datasetName is set somewhere in your app
  useEffect(()=>{
    const storedData = localStorage.getItem('selectedDataset');
    if (storedData) {
      const parsedData = JSON.parse(storedData);
      setDatasetId(parsedData.id);
      setDatasetName(parsedData.name);
    }
  },[]);
  
  // Add to log function
  const addLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setLog(prev => [...prev, `[${timestamp}] ${message}`]);
  };
  
  // Start training function
  const startTraining = async () => {
    const API = (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000').replace(/\/?$/, '/');
    if (!datasetId) {
      addLog('Error: No dataset selected');
      return;
    }
    
    try {
      const response = await fetch(`${API}api/train/${datasetId}/`, {
        method: 'POST',
      });
      
      if (!response.ok) {
        throw new Error('Failed to start training');
      }
      
      const data = await response.json();
      addLog(`Training started: ${data.message}`);
    } catch (error) {
      addLog(`Error starting training: ${error instanceof Error ? error.message : String(error)}`);
    }
  };
  
  // Connect to SSE stream
  useEffect(() => {
    const eventSource = new EventSource(`${API}api/stream/`);
    
    eventSource.onopen = () => {
      setConnected(true);
      addLog('Connected to SSE stream');
    };
    
    eventSource.onerror = (error) => {
      setConnected(false);
      addLog('SSE connection error. Reconnecting...');
      // The browser will automatically try to reconnect
    };
    
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as TrainingStep;
        
        setSteps(prev => ({
          ...prev,
          [data.id]: data
        }));
        
        // Log the step change
        const statusEmoji = data.status === 'completed' ? '✅' : 
                          data.status === 'processing' ? '⏳' :
                          data.status === 'error' ? '❌' : '⏱️';
        
        addLog(`${statusEmoji} Step ${data.id} (${data.name}): ${data.status}`);
        
        // Log details if present
        if (data.details && Object.keys(data.details).length > 0) {
          const detailsStr = Object.entries(data.details)
            .map(([key, value]) => `${key}: ${value}`)
            .join(', ');
          
          if (detailsStr) {
            addLog(`   Details: ${detailsStr}`);
          }
        }
      } catch (error) {
        addLog(`Error parsing SSE data: ${error instanceof Error ? error.message : String(error)}`);
      }
    };
    
    // Cleanup function
    return () => {
      eventSource.close();
      addLog('Disconnected from SSE stream');
    };
  }, []);
  
  // Create ordered array of steps
  const orderedSteps = Object.values(steps).sort((a, b) => a.id - b.id);
  
  // Calculate overall progress
  const completedSteps = orderedSteps.filter(step => step.status === 'completed').length;
  const totalSteps = orderedSteps.length || 9; // Fallback to 9 if no steps received yet
  const progress = totalSteps > 0 ? (completedSteps / totalSteps) * 100 : 0;
  
  return (
    <div className={styles.trainingContainer}>
      <h2 className={styles.title}>Model Training Pipeline</h2>
      
      <div className={styles.connectionStatus}>
        Status: {connected ? (
          <span className={styles.connected}>Connected</span>
        ) : (
          <span className={styles.disconnected}>Disconnected</span>
        )}
      </div>
      
      <button 
        className={styles.startButton} 
        onClick={startTraining}
        disabled={orderedSteps.some(step => step.status === 'processing')}
      >
        Start Training
      </button>
      
      <div className={styles.progressContainer}>
        <div className={styles.overallProgress}>
          <h3>Overall Progress</h3>
          <div className={styles.progressBar}>
            <div 
              className={styles.progressFill} 
              style={{ width: `${progress}%` }}
            />
            <span className={styles.progressText}>
              {Math.round(progress)}% ({completedSteps}/{totalSteps})
            </span>
          </div>
        </div>
        
        <div className={styles.stepsContainer}>
          <h3>Pipeline Steps</h3>
          
          {orderedSteps.length === 0 ? (
            <div className={styles.noSteps}>
              Waiting for training to start...
            </div>
          ) : (
            <ul className={styles.stepsList}>
              {orderedSteps.map((step) => (
                <li 
                  key={step.id} 
                  className={`${styles.step} ${styles[step.status]}`}
                >
                  <div className={styles.stepHeader}>
                    <span className={styles.stepNumber}>{step.id}</span>
                    <span className={styles.stepName}>{step.name}</span>
                    <span className={styles.stepStatus}>
                      {step.status === 'pending' && '⏱️'}
                      {step.status === 'processing' && '⏳'}
                      {step.status === 'completed' && '✅'}
                      {step.status === 'error' && '❌'}
                    </span>
                  </div>
                  
                  {step.status === 'processing' && step.details && step.details.progress !== undefined && (
                    <div className={styles.stepProgressBar}>
                      <div 
                        className={styles.stepProgressFill} 
                        style={{ width: `${step.details.progress}%` }}
                      />
                      <span className={styles.stepProgressText}>
                        {Math.round(step.details.progress)}%
                      </span>
                    </div>
                  )}
                  
                  {step.details && step.details.message && (
                    <div className={styles.stepMessage}>
                      {step.details.message}
                    </div>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
      
      <div className={styles.logContainer}>
        <h3>Training Log</h3>
        <div className={styles.log}>
          {log.map((entry, index) => (
            <pre key={index} className={styles.logEntry}>
              {entry}
            </pre>
          ))}
        </div>
      </div>
    </div>
  );
};

export default TrainingProgress;