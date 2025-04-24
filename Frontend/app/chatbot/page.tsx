
import { ChatInterface } from '@/components/chatbot/chat-interface';
import { ParticlesBackground } from '@/components/ui/particles-background';


export default function ChatbotPage() {
  return (
    <main className="min-h-screen flex flex-col relative">
      <ParticlesBackground />
      <div className="flex-1 container mx-auto py-6">
        <ChatInterface />
      </div>
    </main>
  );
}

