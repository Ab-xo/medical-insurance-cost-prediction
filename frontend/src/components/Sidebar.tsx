"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import {
  Activity,
  BarChart2,
  Home,
  PieChart,
  Info,
  Database,
  DollarSign,
  Menu,
  X,
  HeartPulse,
} from "lucide-react";
import { cn } from "@/lib/utils";

const navItems = [
  { href: "/",        label: "Home",             icon: Home,       desc: "Overview & KPIs" },
  { href: "/dataset", label: "Dataset Overview", icon: Database,   desc: "Data exploration" },
  { href: "/eda",     label: "EDA",              icon: PieChart,   desc: "Visualisations" },
  { href: "/metrics", label: "Model Comparison", icon: BarChart2,  desc: "All 10 models" },
  { href: "/predict", label: "Predict Price",    icon: DollarSign, desc: "Live prediction" },
  { href: "/about",   label: "About",            icon: Info,       desc: "Project info" },
];

function SidebarContent({ onClose }: { onClose?: () => void }) {
  const pathname = usePathname();

  return (
    <div className="flex flex-col h-full">
      {/* Logo */}
      <div className="flex items-center justify-between mb-8 px-2">
        <div className="flex items-center gap-3">
          <div className="bg-primary/20 p-2 rounded-xl">
            <HeartPulse size={22} className="text-primary" />
          </div>
          <div>
            <h1 className="font-bold text-base leading-tight tracking-tight">
              MedPredict<span className="text-primary">.ai</span>
            </h1>
            <p className="text-[10px] text-muted-foreground leading-tight">Insurance ML Dashboard</p>
          </div>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className="md:hidden p-1.5 rounded-lg hover:bg-secondary/50 text-muted-foreground"
          >
            <X size={18} />
          </button>
        )}
      </div>

      {/* Nav label */}
      <p className="text-[10px] uppercase tracking-widest text-muted-foreground font-semibold px-3 mb-2">
        Navigation
      </p>

      {/* Nav links */}
      <nav className="flex flex-col gap-1 flex-1">
        {navItems.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              onClick={onClose}
              className={cn(
                "group flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-200",
                isActive
                  ? "bg-primary text-primary-foreground shadow-md"
                  : "text-muted-foreground hover:bg-secondary/60 hover:text-foreground"
              )}
            >
              <item.icon
                size={17}
                className={cn(
                  "shrink-0 transition-transform duration-200 group-hover:scale-110",
                  isActive && "scale-110"
                )}
              />
              <div className="min-w-0">
                <p className="truncate leading-tight">{item.label}</p>
                <p
                  className={cn(
                    "text-[10px] leading-tight truncate",
                    isActive ? "text-primary-foreground/70" : "text-muted-foreground/60"
                  )}
                >
                  {item.desc}
                </p>
              </div>
            </Link>
          );
        })}
      </nav>

      {/* Footer status */}
      <div className="mt-6 px-1">
        <div className="bg-secondary/30 rounded-xl p-3 text-xs text-muted-foreground border border-border/50 space-y-2">
          <p className="font-semibold text-foreground text-sm flex items-center gap-2">
            <Activity size={13} className="text-primary" /> System Status
          </p>
          <p className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            API: <span className="text-green-400 font-medium">localhost:8000</span>
          </p>
          <p className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-blue-400" />
            Best model: <span className="text-blue-400 font-medium">Lasso</span>
          </p>
        </div>
      </div>
    </div>
  );
}

export function Sidebar() {
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <>
      {/* Desktop sidebar */}
      <aside className="w-64 h-screen fixed top-0 left-0 border-r border-border/60 bg-card/60 backdrop-blur-xl hidden md:flex flex-col px-4 py-7 z-50">
        <SidebarContent />
      </aside>

      {/* Mobile top bar */}
      <div className="md:hidden fixed top-0 left-0 right-0 z-50 bg-card/80 backdrop-blur-md border-b border-border/60 flex items-center justify-between px-4 h-14">
        <div className="flex items-center gap-2">
          <HeartPulse size={20} className="text-primary" />
          <span className="font-bold text-sm">
            MedPredict<span className="text-primary">.ai</span>
          </span>
        </div>
        <button
          onClick={() => setMobileOpen(true)}
          className="p-2 rounded-lg hover:bg-secondary/50"
        >
          <Menu size={20} />
        </button>
      </div>

      {/* Mobile drawer overlay */}
      {mobileOpen && (
        <div
          className="md:hidden fixed inset-0 z-50 bg-black/60 backdrop-blur-sm"
          onClick={() => setMobileOpen(false)}
        >
          <div
            className="absolute top-0 left-0 w-72 h-full bg-card border-r border-border/60 px-4 py-7 overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <SidebarContent onClose={() => setMobileOpen(false)} />
          </div>
        </div>
      )}
    </>
  );
}
