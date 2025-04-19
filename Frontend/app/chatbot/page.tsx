import Link from 'next/link';
import { BrainCircuit, ChevronLeft } from 'lucide-react';
import { ChatInterface } from '@/components/chatbot/chat-interface';
import { ParticlesBackground } from '@/components/ui/particles-background';
import { buttonVariants } from '@/components/ui/button';

export default function ChatbotPage() {
  return (
    <main className="min-h-screen flex flex-col relative">
      <ParticlesBackground />
      
      <header className="border-b border-border/40 backdrop-blur-sm sticky top-0 z-10">
        <div className="container mx-auto flex items-center justify-between h-16 px-4">
          <div className="flex items-center gap-2">
            <Link 
              href="/" 
              className={buttonVariants({
                variant: "ghost",
                size: "icon",
                className: "rounded-full"
              })}
            >
              <ChevronLeft className="h-5 w-5" />
              <span className="sr-only">Back to Home</span>
            </Link>
            
            <div className="flex items-center gap-2">
              <BrainCircuit className="h-5 w-5 text-chart-1" />
              <span className="font-medium">NEORO PILOT</span>
            </div>
          </div>
        </div>
      </header>
      
      <div className="flex-1 container mx-auto py-6">
        <ChatInterface />
      </div>
    </main>
  );
}