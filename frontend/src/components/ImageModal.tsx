"use client";

import { useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Minimize2, ChevronLeft, ChevronRight } from "lucide-react";

export interface ModalImage {
  url: string;
  label: string;
}

interface Props {
  open: boolean;
  image: ModalImage | null;
  /** Pass siblings to enable prev/next navigation. Leave undefined to hide arrows. */
  siblings?: ModalImage[];
  /** Index of the current image inside siblings */
  siblingIndex?: number;
  onClose: () => void;
  onMinimize: () => void;
  onNavigate?: (index: number) => void;
}

export function ImageModal({
  open,
  image,
  siblings,
  siblingIndex = 0,
  onClose,
  onMinimize,
  onNavigate,
}: Props) {
  const hasSiblings = siblings && siblings.length > 1 && onNavigate;

  const prev = useCallback(() => {
    if (!hasSiblings || siblings == null) return;
    onNavigate!((siblingIndex - 1 + siblings.length) % siblings.length);
  }, [hasSiblings, siblings, siblingIndex, onNavigate]);

  const next = useCallback(() => {
    if (!hasSiblings || siblings == null) return;
    onNavigate!((siblingIndex + 1) % siblings.length);
  }, [hasSiblings, siblings, siblingIndex, onNavigate]);

  // keyboard support
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape")      onClose();
      if (e.key === "ArrowLeft")   prev();
      if (e.key === "ArrowRight")  next();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose, prev, next]);

  return (
    <AnimatePresence>
      {open && image && (
        /* backdrop */
        <motion.div
          key="modal-backdrop"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="fixed inset-0 z-[200] flex items-center justify-center bg-black/85 backdrop-blur-sm p-4"
          onClick={onClose}
        >
          {/* panel */}
          <motion.div
            key="modal-panel"
            initial={{ scale: 0.92, opacity: 0 }}
            animate={{ scale: 1,    opacity: 1 }}
            exit={{ scale: 0.92,    opacity: 0 }}
            transition={{ type: "spring", stiffness: 280, damping: 26 }}
            className="relative w-full max-w-5xl bg-card border border-border/60 rounded-2xl shadow-2xl overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {/* ── toolbar ── */}
            <div className="flex items-center justify-between px-4 py-2.5 border-b border-border/50 bg-secondary/20 backdrop-blur-md">
              <span className="text-sm font-medium truncate max-w-[60%]">{image.label}</span>

              <div className="flex items-center gap-1">
                {hasSiblings && (
                  <>
                    <button
                      onClick={prev}
                      className="p-1.5 rounded-lg hover:bg-secondary/60 transition-colors text-muted-foreground hover:text-foreground"
                      title="Previous (←)"
                    >
                      <ChevronLeft size={16} />
                    </button>
                    <span className="text-xs text-muted-foreground px-1 tabular-nums">
                      {siblingIndex + 1} / {siblings!.length}
                    </span>
                    <button
                      onClick={next}
                      className="p-1.5 rounded-lg hover:bg-secondary/60 transition-colors text-muted-foreground hover:text-foreground"
                      title="Next (→)"
                    >
                      <ChevronRight size={16} />
                    </button>
                    <div className="w-px h-4 bg-border/60 mx-1" />
                  </>
                )}

                {/* minimize — collapses modal back to card */}
                <button
                  onClick={onMinimize}
                  className="p-1.5 rounded-lg hover:bg-secondary/60 transition-colors text-muted-foreground hover:text-foreground"
                  title="Minimize"
                >
                  <Minimize2 size={15} />
                </button>

                {/* close */}
                <button
                  onClick={onClose}
                  className="p-1.5 rounded-lg hover:bg-destructive/20 hover:text-destructive transition-colors text-muted-foreground"
                  title="Close (Esc)"
                >
                  <X size={15} />
                </button>
              </div>
            </div>

            {/* ── image ── */}
            <div className="bg-white/[0.03] flex items-center justify-center p-4 md:p-6"
                 style={{ minHeight: "60vh", maxHeight: "82vh" }}>
              <AnimatePresence mode="wait">
                <motion.img
                  key={image.url}
                  src={image.url}
                  alt={image.label}
                  initial={{ opacity: 0, scale: 0.97 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.97 }}
                  transition={{ duration: 0.18 }}
                  className="max-w-full max-h-[75vh] object-contain rounded-lg select-none"
                  // eslint-disable-next-line @next/next/no-img-element
                />
              </AnimatePresence>
            </div>

            {/* ── bottom hint ── */}
            <div className="px-4 py-2 bg-secondary/10 border-t border-border/40 flex items-center justify-between text-[11px] text-muted-foreground/60">
              <span>Click outside or press Esc to close</span>
              {hasSiblings && <span>Use ← → arrow keys to navigate</span>}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
