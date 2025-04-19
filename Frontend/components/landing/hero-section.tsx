'use client';

import Link from 'next/link';
import { buttonVariants } from '@/components/ui/button';
import { AnimatedGradient } from '@/components/ui/animated-gradient';
import { BarChart3, BrainCircuit, Microscope, Network, Zap } from 'lucide-react';
import { motion } from 'framer-motion';

export function HeroSection() {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
        delayChildren: 0.3,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.6,
        ease: "easeOut",
      },
    },
  };

  return (
    <section className="relative min-h-[90vh] flex items-center overflow-hidden neural-grid">
      <AnimatedGradient />
      
      <div className="absolute inset-0 hexagon-pattern opacity-20" />
      
      <motion.div 
        className="container px-4 mx-auto relative z-10"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        <div className="flex flex-col items-center text-center max-w-4xl mx-auto">
          <motion.div 
            variants={itemVariants}
            className="inline-flex items-center gap-2 glass-effect px-6 py-3 rounded-full mb-8"
          >
            <Network className="w-5 h-5 text-chart-1 animate-pulse-slow" />
            <span className="text-sm font-medium bg-clip-text text-transparent bg-gradient-to-r from-chart-1 to-chart-2">
              Auto Machine Learning Pipeline
            </span>
          </motion.div>
          
          <motion.h1 
            variants={itemVariants}
            className="text-4xl md:text-5xl lg:text-7xl font-bold tracking-tight mb-6 text-glow"
          >
            <span className="background-animate bg-clip-text text-transparent bg-gradient-to-r from-primary via-chart-1 to-chart-2">
              NEORO PILOT
            </span>
          </motion.h1>
          
          <motion.p 
            variants={itemVariants}
            className="text-xl md:text-2xl text-muted-foreground mb-10 max-w-2xl leading-relaxed"
          >
            Revolutionize your ML workflow with AI-powered automation. Build, train, and deploy 
            production-ready models without writing a single line of code.
          </motion.p>
          
          <motion.div variants={itemVariants}>
            <Link 
              href="/chatbot" 
              className={buttonVariants({
                size: "lg",
                className: "px-8 py-6 text-lg group transition-all duration-300 animate-shimmer bg-[linear-gradient(110deg,#000103,45%,#1e293b,55%,#000103)] bg-[length:200%_100%] border-glow"
              })}
            >
              <span>Launch AutoML Studio</span>
              <Zap className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Link>
          </motion.div>
          
          <motion.div 
            variants={containerVariants}
            className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-24"
          >
            <motion.div 
              variants={itemVariants}
              whileHover={{ scale: 1.05, transition: { duration: 0.2 } }}
              className="glass-effect p-8 rounded-xl flex flex-col items-center text-center group hover:border-chart-1/30 transition-colors"
            >
              <div className="w-14 h-14 rounded-full bg-chart-1/10 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                <Zap className="w-7 h-7 text-chart-1" />
              </div>
              <h3 className="text-xl font-medium mb-3 text-glow">Neural Architecture Search</h3>
              <p className="text-muted-foreground">Automated model architecture optimization using state-of-the-art AI techniques</p>
            </motion.div>
            
            <motion.div 
              variants={itemVariants}
              whileHover={{ scale: 1.05, transition: { duration: 0.2 } }}
              className="glass-effect p-8 rounded-xl flex flex-col items-center text-center group hover:border-chart-2/30 transition-colors"
            >
              <div className="w-14 h-14 rounded-full bg-chart-2/10 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                <Microscope className="w-7 h-7 text-chart-2" />
              </div>
              <h3 className="text-xl font-medium mb-3 text-glow">Hyperparameter Optimization</h3>
              <p className="text-muted-foreground">Advanced Bayesian optimization for finding optimal model configurations</p>
            </motion.div>
            
            <motion.div 
              variants={itemVariants}
              whileHover={{ scale: 1.05, transition: { duration: 0.2 } }}
              className="glass-effect p-8 rounded-xl flex flex-col items-center text-center group hover:border-chart-4/30 transition-colors"
            >
              <div className="w-14 h-14 rounded-full bg-chart-4/10 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                <BarChart3 className="w-7 h-7 text-chart-4" />
              </div>
              <h3 className="text-xl font-medium mb-3 text-glow">Automated Feature Engineering</h3>
              <p className="text-muted-foreground">Intelligent feature selection and transformation using deep learning</p>
            </motion.div>
          </motion.div>
        </div>
      </motion.div>
    </section>
  );
}