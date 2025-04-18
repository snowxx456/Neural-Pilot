import React, { useState } from 'react';
import { motion } from 'framer-motion';
import ChatInterface from '../components/ChatInterface';
import { HiArrowRight, HiOutlineLightningBolt, HiOutlineChip } from 'react-icons/hi';

export default function Landing() {
  const [showChatInterface, setShowChatInterface] = useState(false);

  const handleStartClick = () => {
    setShowChatInterface(true);
  };

  return (
    <main className="min-h-screen grid-pattern flex flex-col">
      {/* Animated background elements */}
      <div className="fixed inset-0 z-0">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 rounded-full bg-neo-primary opacity-5 blur-3xl animate-pulse-slow"></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 rounded-full bg-neo-secondary opacity-5 blur-3xl animate-pulse-slow" style={{ animationDelay: '2s' }}></div>
      </div>

      {/* Header */}
      <header className="relative z-10 p-6 flex justify-between items-center">
        <div className="flex items-center">
          <img src="/images/neoro-logo.svg" alt="NEORO PILOT Logo" className="h-10" />
        </div>
        <nav>
          <ul className="flex gap-8">
            <li><a href="#" className="text-neo-light hover:text-white transition-colors">Docs</a></li>
            <li><a href="#" className="text-neo-light hover:text-white transition-colors">Examples</a></li>
            <li><a href="#" className="text-neo-light hover:text-white transition-colors">GitHub</a></li>
          </ul>
        </nav>
      </header>

      {/* Main content */}
      <div className="relative z-10 flex-grow flex flex-col items-center justify-center px-6 -mt-20">
        {!showChatInterface ? (
          <div className="text-center max-w-3xl">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="mb-8"
            >
              <h1 className="text-5xl font-bold mb-4 text-white">
                <span className="neo-gradient-text">NEORO PILOT</span>
              </h1>
              <h2 className="text-3xl font-medium mb-2 text-white">Auto Machine Learning Pipeline</h2>
              <p className="text-xl text-neo-light">Empowering your ML journey in one click</p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.6, delay: 0.3 }}
              className="flex justify-center mb-12"
            >
              <img src="/images/tech-illustration.svg" alt="AI Illustration" className="w-80 h-80 animate-float" />
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.6 }}
              className="flex flex-col items-center"
            >
              <button
                onClick={handleStartClick}
                className="bg-neo-primary hover:bg-opacity-90 text-white py-3 px-8 rounded-lg flex items-center gap-2 transition-all mb-6"
              >
                Try Our AutoML Application <HiArrowRight />
              </button>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
                <div className="glass-effect p-6 rounded-lg">
                  <div className="bg-neo-primary bg-opacity-20 w-12 h-12 rounded-lg flex items-center justify-center mb-4">
                    <HiOutlineLightningBolt className="text-neo-primary text-2xl" />
                  </div>
                  <h3 className="text-white text-lg font-medium mb-2">Lightning Fast</h3>
                  <p className="text-neo-light text-sm">Generate ML models in seconds with our optimized AutoML pipeline</p>
                </div>

                <div className="glass-effect p-6 rounded-lg">
                  <div className="bg-neo-secondary bg-opacity-20 w-12 h-12 rounded-lg flex items-center justify-center mb-4">
                    <HiOutlineChip className="text-neo-secondary text-2xl" />
                  </div>
                  <h3 className="text-white text-lg font-medium mb-2">Smart Algorithms</h3>
                  <p className="text-neo-light text-sm">Backed by intelligent preprocessing and hyperparameter tuning</p>
                </div>

                <div className="glass-effect p-6 rounded-lg">
                  <div className="bg-neo-tertiary bg-opacity-20 w-12 h-12 rounded-lg flex items-center justify-center mb-4">
                    <HiOutlineChip className="text-neo-tertiary text-2xl" />
                  </div>
                  <h3 className="text-white text-lg font-medium mb-2">Deploy Ready</h3>
                  <p className="text-neo-light text-sm">Easily deploy models to production with seamless integration support</p>
                </div>
              </div>
            </motion.div>
          </div>
        ) : (
          <ChatInterface />
        )}
      </div>
    </main>
  );
}
