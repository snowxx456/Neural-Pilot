'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';
import { buttonVariants } from '@/components/ui/button';
import { BrainCircuit, Database, LineChart, Microscope, MessageSquareCode } from 'lucide-react';

const navItems = [
  
  {
    label: 'Preprocessing',
    href: '/preprocessing',
    icon: Microscope
  },
  {
    label: 'Visualization',
    href: '/visualization',
    icon: LineChart,
    
  },
  {
    label: 'Model Training',
    href: '/model_training',
    icon: BrainCircuit,
    disabled: false
  }
];

export function Navbar() {
  const pathname = usePathname();

  return (
    <header className="border-b border-border/40 backdrop-blur-sm sticky top-0 z-50">
      <div className="container mx-auto px-4">
        <div className="h-16 flex items-center justify-between gap-4">
          <Link 
            href="/" 
            className="flex items-center gap-2 font-semibold text-lg"
          >
            <BrainCircuit className="h-5 w-5 text-primary" />
            <span>NEORO PILOT</span>
          </Link>

          <nav className="flex items-center gap-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <Link
                  key={item.href}
                  href={item.disabled ? '#' : item.href}
                  className={cn(
                    buttonVariants({ variant: "ghost", size: "sm" }),
                    "gap-2",
                    pathname === item.href && "bg-primary/10 text-primary",
                    item.disabled && "opacity-50 cursor-not-allowed"
                  )}
                  onClick={(e) => item.disabled && e.preventDefault()}
                >
                  <Icon className="h-4 w-4" />
                  {item.label}
                </Link>
              );
            })}

            <div className="w-px h-6 bg-border/60 mx-2" />

            <Link
              href="/chatbot"
              className={buttonVariants({
                size: "sm",
                className: "gap-2 bg-primary/10 text-primary hover:bg-primary/20"
              })}
            >
              <MessageSquareCode className="h-4 w-4" />
              AI Assistant
            </Link>
          </nav>
        </div>
      </div>
    </header>
  );
}