'use client';

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { PreprocessingTab } from "@/components/preprocessing/preprocessing-tab";
import { BrainCircuit } from 'lucide-react';

export function PreprocessingWrapper() {
  const handleVisualize = () => {
    console.log('Visualize');
  };

  return (
    <div className="glass-effect rounded-2xl border border-border/50 overflow-hidden">
      <Tabs defaultValue="preprocessing" className="w-full">
        <TabsList className="w-full border-b border-border/50 p-2 bg-background/50">
          <TabsTrigger 
            value="preprocessing"
            className="gap-2 data-[state=active]:bg-primary/20 data-[state=active]:text-primary"
          >
            <BrainCircuit className="w-4 h-4" />
            Data Preprocessing
          </TabsTrigger>
        </TabsList>
        <TabsContent value="preprocessing" className="p-0">
          <PreprocessingTab 
            datasetName="Iris Classification Dataset"
            onVisualize={handleVisualize}
          />
        </TabsContent>
      </Tabs>
    </div>
  );
}