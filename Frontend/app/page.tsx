import { HeroSection } from '@/components/landing/hero-section';
import { ParticlesBackground } from '@/components/ui/particles-background';

export default function Home() {
  return (
    <main className="min-h-screen flex flex-col relative overflow-hidden">
      <ParticlesBackground />
      <HeroSection />
    </main>
  );
}