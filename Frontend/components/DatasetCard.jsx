import React from 'react';
import { motion } from 'framer-motion';
import { HiDownload, HiExternalLink } from 'react-icons/hi';

const DatasetCard = ({ dataset }) => {
  const { title, description, downloadUrl, externalLink } = dataset;
  
  return (
    <motion.div 
      className="dataset-card glass-effect rounded-lg overflow-hidden p-5 w-72 flex flex-col"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      whileHover={{ scale: 1.03 }}
    >
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-lg font-bold text-white">{title}</h3>
        <div className="h-8 w-8 rounded-full bg-neo-primary bg-opacity-20 flex items-center justify-center">
          <span className="text-neo-primary text-sm">DS</span>
        </div>
      </div>
      
      <p className="text-neo-light text-sm mb-4 flex-grow">{description}</p>
      
      <div className="flex flex-col gap-2 mt-2">
        <a 
          href={downloadUrl}
          className="flex items-center justify-center gap-2 py-2 px-4 bg-neo-primary bg-opacity-20 hover:bg-opacity-30 rounded-md text-white text-sm transition-all"
        >
          <HiDownload className="text-neo-primary" />
          Download Dataset
        </a>
        
        <a 
          href={externalLink}
          className="flex items-center justify-center gap-2 py-2 px-4 bg-neo-dark hover:bg-opacity-80 rounded-md text-neo-light text-sm transition-all"
          target="_blank"
          rel="noopener noreferrer"
        >
          <HiExternalLink className="text-neo-secondary" />
          View Source
        </a>
      </div>
    </motion.div>
  );
};

export default DatasetCard;