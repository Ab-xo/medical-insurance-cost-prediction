"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ImageModal, type ModalImage } from "@/components/ImageModal";
import { Maximize2, Minimize2 } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface Props {
  label: string;
  url: string;
  icon?: React.ReactNode;
  /** Pass the full sibling list + this image's index to enable in-modal prev/next */
  siblings?: ModalImage[];
  siblingIndex?: number;
  /** Extra classes applied to the outer Card */
  className?: string;
  /** Default collapsed (minimized) state */
  defaultMinimized?: boolean;
}

export function ChartCard({
  label,
  url,
  icon,
  siblings,
  siblingIndex = 0,
  className = "",
  defaultMinimized = false,
}: Props) {
  const [minimized, setMinimized] = useState(defaultMinimized);
  const [modalOpen, setModalOpen] = useState(false);

  // When the modal navigates, update the modal's current image via the siblings list
  const [modalIndex, setModalIndex] = useState(siblingIndex);

  // Sync modalIndex whenever the card's own siblingIndex changes from parent
  const currentModalImage = siblings ? siblings[modalIndex] : { url, label };

  function openModal() {
    setModalIndex(siblingIndex);
    setModalOpen(true);
  }

  return (
    <>
      <Card
        className={`bg-card/40 backdrop-blur-md border-primary/20 overflow-hidden transition-all duration-300 ${className}`}
      >
        {/* ── header ── */}
        <CardHeader className="flex flex-row items-center justify-between py-2.5 px-3 border-b border-border/40 bg-secondary/10">
          <CardTitle className="text-xs font-medium flex items-center gap-1.5 truncate">
            {icon && <span className="text-primary shrink-0">{icon}</span>}
            <span className="truncate">{label}</span>
          </CardTitle>

          <div className="flex items-center gap-0.5 shrink-0 ml-2">
            {/* minimize / restore */}
            <button
              onClick={() => setMinimized((v) => !v)}
              className="p-1.5 rounded-md hover:bg-secondary/60 text-muted-foreground hover:text-foreground transition-colors"
              title={minimized ? "Restore" : "Minimize"}
            >
              <Minimize2 size={13} />
            </button>

            {/* maximize → open modal */}
            <button
              onClick={openModal}
              className="p-1.5 rounded-md hover:bg-secondary/60 text-muted-foreground hover:text-foreground transition-colors"
              title="Maximize"
            >
              <Maximize2 size={13} />
            </button>
          </div>
        </CardHeader>

        {/* ── body (collapses when minimized) ── */}
        <AnimatePresence initial={false}>
          {!minimized && (
            <motion.div
              key="body"
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.22, ease: "easeInOut" }}
              className="overflow-hidden"
            >
              <CardContent
                className="p-0 bg-white/[0.03] cursor-zoom-in"
                onClick={openModal}
              >
                {/* Fixed-height container so all cards are equal height in the grid */}
                <div className="h-56 flex items-center justify-center p-2">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={url}
                    alt={label}
                    className="h-full w-full object-contain hover:scale-[1.02] transition-transform duration-300"
                    loading="lazy"
                  />
                </div>
              </CardContent>
            </motion.div>
          )}
        </AnimatePresence>
      </Card>

      {/* ── modal ── */}
      <ImageModal
        open={modalOpen}
        image={currentModalImage}
        siblings={siblings}
        siblingIndex={modalIndex}
        onClose={() => setModalOpen(false)}
        onMinimize={() => setModalOpen(false)}
        onNavigate={(i) => setModalIndex(i)}
      />
    </>
  );
}
