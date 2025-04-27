"use client";

import React from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface DataTableProps {
  data: any[];
}

export function DataTable({ data }: DataTableProps) {
  if (!data.length) return null;

  const columns = Object.keys(data[0]);

  return (
    <div className="w-full overflow-x-auto max-w-[calc(100vw-4rem)]">
      <table className="w-full table-auto">
        <thead>
          <tr className="border-b border-border/50 bg-card/30">
            {columns.map((column) => (
              <th
                key={column}
                className="sticky top-0 px-4 py-3 text-left text-sm font-medium text-muted-foreground whitespace-nowrap bg-card/30"
              >
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, rowIndex) => (
            <motion.tr
              key={rowIndex}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: rowIndex * 0.05 }}
              className={cn(
                "border-b border-border/50 transition-colors",
                "hover:bg-card/50"
              )}
            >
              {columns.map((column) => (
                <td
                  key={`${rowIndex}-${column}`}
                  className="px-4 py-3 text-sm whitespace-nowrap"
                >
                  {row[column]}
                </td>
              ))}
            </motion.tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}