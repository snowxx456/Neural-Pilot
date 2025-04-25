"use client";

import React, { useState, useRef, useEffect } from "react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Send, Upload, Sparkles } from "lucide-react";
import { motion } from "framer-motion";
import axios from "axios";
import { toast } from "@/components/ui/use-toast"; // Make sure this import is correct
import { ToastAction } from "../ui/toast";
import { useRouter } from "next/navigation";

interface ChatInputProps {
  onSend: (message: string) => void;
  onFileUpload?: (file: File) => void;
  isProcessing: boolean;
  placeholder?: string;
}

export function ChatInput({
  onSend,
  onFileUpload,
  isProcessing,
  placeholder = "What topic do you want to build a machine learning model on?",
}: ChatInputProps) {
  const [input, setInput] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadingFile, setUploadingFile] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();

  // API URL - use your actual backend URL
  const API_URL =
    process.env.NEXT_PUBLIC_SERVER_URL || "http://localhost:8000/";

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !isProcessing) {
      onSend(input.trim());
      setInput("");
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);

      // If onFileUpload prop exists, call it (for local UI updates)
      if (onFileUpload) {
        onFileUpload(file);
      }

      // Also send the file to the backend
      await uploadFileToBackend(file);

      // Reset the file input
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  // Function to upload file to backend
  const uploadFileToBackend = async (file: File) => {
    try {
      setUploadingFile(true);

      // Create FormData object
      const formData = new FormData();
      formData.append("file", file);

      // Send file to backend
      const response = await axios.post(`${API_URL}api/upload/`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      console.log("File upload response:", response.data);

      // Store dataset information in localStorage
      if (response.data.id && response.data.name) {
        const datasetInfo = {
          id: response.data.id,
          name: response.data.name,
          timestamp: new Date().toISOString(),
        };

        // Replace any existing dataset in local storage
        localStorage.setItem("selectedDataset", JSON.stringify(datasetInfo));

        // Show success toast
        toast({
          title: "Dataset uploaded and selected",
          description: `Your dataset "${response.data.name}" has been uploaded and selected for your workspace.`,

        });
      } else {
        toast({
          title: "File uploaded",
          description: `Your file ${file.name} has been uploaded to the server.`,
        });
      }

      return response.data;
    } catch (error) {
      console.error("Error uploading file:", error);

      // Show error toast
      toast({
        title: "Upload failed",
        description:
          "There was an error uploading your file. Please try again.",
        variant: "destructive",
      });

      throw error;
    } finally {
      setUploadingFile(false);
    }
  };
  const triggerFileUpload = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = "auto";
      inputRef.current.style.height = `${Math.min(
        inputRef.current.scrollHeight,
        200
      )}px`;
    }
  }, [input]);

  return (
    <motion.form
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      onSubmit={handleSubmit}
      className="relative w-full max-w-4xl mx-auto"
    >
      <div className="relative flex items-center">
        {/* File upload button */}
        <Button
          type="button"
          onClick={triggerFileUpload}
          size="icon"
          disabled={uploadingFile}
          className={cn(
            "absolute left-1.5 h-9 w-9 rounded-lg bg-secondary/30 hover:bg-secondary/50 transition-all z-10",
            uploadingFile && "opacity-50 cursor-not-allowed"
          )}
        >
          <Upload className={cn("h-4 w-4", uploadingFile && "animate-pulse")} />
          <span className="sr-only">Upload CSV</span>
        </Button>

        {/* File input element */}
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept=".csv"
          className="hidden"
          id="file-upload"
          disabled={uploadingFile}
        />

        {/* Selected file indicator */}
        {/*selectedFile && (
          <div className="absolute left-12 top-1/2 -translate-y-1/2 text-sm text-muted-foreground ml-2 truncate max-w-[150px]">
            {selectedFile.name}
          </div>
        )*/}

        <textarea
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={uploadingFile ? "Uploading file..." : placeholder}
          disabled={isProcessing || uploadingFile}
          rows={1}
          className={cn(
            "resize-none w-full rounded-xl py-3",
            "pl-12",
            "pr-14",
            "glass-effect focus:ring-1 focus:ring-primary",
            "text-sm transition-all",
            "focus:outline-none",
            "overflow-hidden",
            (isProcessing || uploadingFile) && "opacity-70"
          )}
        />

        <Button
          type="submit"
          size="icon"
          disabled={!input.trim() || isProcessing || uploadingFile}
          className={cn(
            "absolute right-1.5 h-9 w-9 rounded-lg",
            "bg-primary hover:bg-primary/90",
            "transition-all z-10",
            (!input.trim() || isProcessing || uploadingFile) && "opacity-70"
          )}
        >
          <Send className="h-4 w-4" />
          <span className="sr-only">Send</span>
        </Button>
      </div>
    </motion.form>
  );
}
