'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState } from 'react';
import { cn } from '@/lib/utils';
import { buttonVariants } from '@/components/ui/button';
import { BrainCircuit, LineChart, Microscope, MessageSquareCode, Menu, X } from 'lucide-react';

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
  const [isMenuOpen, setIsMenuOpen] = useState(false);

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

          {/* Mobile menu button */}
          <button
            className="lg:hidden"
            onClick={() => setIsMenuOpen(!isMenuOpen)}
          >
            {isMenuOpen ? (
              <X className="h-6 w-6" />
            ) : (
              <Menu className="h-6 w-6" />
            )}
          </button>

          {/* Desktop navigation */}
          <nav className="hidden lg:flex items-center gap-1">
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

        {/* Mobile navigation */}
        {isMenuOpen && (
          <nav className="lg:hidden py-4 flex flex-col gap-2">
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <Link
                  key={item.href}
                  href={item.disabled ? '#' : item.href}
                  className={cn(
                    buttonVariants({ variant: "ghost", size: "sm" }),
                    "w-full justify-start gap-2",
                    pathname === item.href && "bg-primary/10 text-primary",
                    item.disabled && "opacity-50 cursor-not-allowed"
                  )}
                  onClick={(e) => {
                    if (item.disabled) e.preventDefault();
                    setIsMenuOpen(false);
                  }}
                >
                  <Icon className="h-4 w-4" />
                  {item.label}
                </Link>
              );
            })}

            <div className="h-px w-full bg-border/60 my-2" />

            <Link
              href="/chatbot"
              className={buttonVariants({
                size: "sm",
                className: "w-full justify-start gap-2 bg-primary/10 text-primary hover:bg-primary/20"
              })}
              onClick={() => setIsMenuOpen(false)}
            >
              <MessageSquareCode className="h-4 w-4" />
              AI Assistant
            </Link>
          </nav>
        )}
      </div>
    </header>
  );
}