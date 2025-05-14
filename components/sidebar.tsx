"use client"

import { useState } from "react"
import { Plus, FileSpreadsheet } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

interface SidebarProps {
  activeSheet: string
  sheets: string[]
  onSelectSheet: (sheet: string) => void
  onCreateSheet: (name: string) => void
}

export function Sidebar({ activeSheet, sheets, onSelectSheet, onCreateSheet }: SidebarProps) {
  const [showNewSheetDialog, setShowNewSheetDialog] = useState(false)
  const [newSheetName, setNewSheetName] = useState("")

  const handleCreateSheet = () => {
    if (newSheetName.trim()) {
      onCreateSheet(newSheetName)
      setNewSheetName("")
      setShowNewSheetDialog(false)
    }
  }

  return (
    <div className="w-64 border-r bg-muted/10">
      <div className="flex h-12 items-center justify-between border-b px-4">
        <h2 className="text-sm font-medium">Sheets</h2>
        <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => setShowNewSheetDialog(true)}>
          <Plus className="h-4 w-4" />
          <span className="sr-only">New sheet</span>
        </Button>
      </div>
      <div className="p-2">
        {sheets.map((sheet) => (
          <button
            key={sheet}
            className={`flex w-full items-center gap-2 rounded-md px-3 py-2 text-left text-sm ${
              activeSheet === sheet ? "bg-accent text-accent-foreground" : "text-muted-foreground hover:bg-accent/50"
            }`}
            onClick={() => onSelectSheet(sheet)}
          >
            <FileSpreadsheet className="h-4 w-4" />
            {sheet}
          </button>
        ))}
      </div>

      {/* New Sheet Dialog */}
      <Dialog open={showNewSheetDialog} onOpenChange={setShowNewSheetDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create New Sheet</DialogTitle>
            <DialogDescription>Enter a name for your new sheet.</DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <Label htmlFor="sheet-name">Sheet Name</Label>
              <Input
                id="sheet-name"
                placeholder="e.g., Customer Data, Project Tasks"
                value={newSheetName}
                onChange={(e) => setNewSheetName(e.target.value)}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowNewSheetDialog(false)}>
              Cancel
            </Button>
            <Button onClick={handleCreateSheet}>Create Sheet</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
