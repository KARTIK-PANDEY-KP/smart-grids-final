"use client"

import { useState, useEffect } from "react"
import { PlusCircle, Save, Download, Settings, User, LogOut, Plus, Trash2, Loader2, RefreshCw, Database } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Label } from "@/components/ui/label"
import { Sidebar } from "@/components/sidebar"

// Define types for sheet data structure
interface SheetData {
  columns: string[];
  rows: string[][];
  aiColumns: Record<string, boolean>;
}

interface SheetStore {
  [key: string]: SheetData;
}

// Sample data for the spreadsheet
const initialSheets: SheetStore = {
  "Student Info": {
    columns: ["First Name", "Last Name", "Major", "GPA"],
    rows: [
      ["John", "Doe", "Computer Science", "3.8"],
      ["Jane", "Smith", "Electrical Engineering", "3.9"],
      ["Bob", "Johnson", "Psychology", "3.5"],
      ["Alice", "Williams", "Biology", "3.7"],
      ["Charlie", "Brown", "Mathematics", "3.6"],
    ],
    aiColumns: {},
  },
  Leads: {
    columns: ["Company", "Contact", "Email", "Industry"],
    rows: [
      ["Acme Inc", "John Smith", "john@acme.com", "Technology"],
      ["XYZ Corp", "Jane Doe", "jane@xyz.com", "Healthcare"],
      ["ABC Ltd", "Bob Johnson", "bob@abc.com", "Finance"],
      ["123 Co", "Alice Williams", "alice@123.co", "Retail"],
      ["Tech Solutions", "Charlie Brown", "charlie@techsolutions.com", "Technology"],
    ],
    aiColumns: {},
  },
}

export default function AppPage() {
  const [activeSheet, setActiveSheet] = useState("Student Info")
  const [sheets, setSheets] = useState<SheetStore>(initialSheets)
  const [showAIDialog, setShowAIDialog] = useState(false)
  const [aiPrompt, setAiPrompt] = useState("")
  const [aiColumnName, setAiColumnName] = useState("")
  const [loadingRows, setLoadingRows] = useState<number[]>([])
  const [selectedRows, setSelectedRows] = useState<number[]>([])
  const [lastAiPrompt, setLastAiPrompt] = useState("")
  const [dbSaveStatus, setDbSaveStatus] = useState<string | null>(null)

  // Load sheets from localStorage on component mount
  useEffect(() => {
    // Don't load from localStorage on first mount - use initial data
    // const savedSheets = localStorage.getItem("smartgrid-sheets")
    // if (savedSheets) {
    //   setSheets(JSON.parse(savedSheets))
    // }
    
    // Optional: clear localStorage to ensure clean state each time
    localStorage.removeItem("smartgrid-sheets")
  }, [])

  // Save sheets to localStorage when they change
  useEffect(() => {
    localStorage.setItem("smartgrid-sheets", JSON.stringify(sheets))
  }, [sheets])

  const currentSheet = sheets[activeSheet]

  const addRow = () => {
    const newSheets = { ...sheets }
    const emptyRow = Array(currentSheet.columns.length).fill("")
    newSheets[activeSheet].rows.push(emptyRow)
    setSheets(newSheets)
  }

  const deleteRow = (rowIndex: number) => {
    const newSheets = { ...sheets }
    newSheets[activeSheet].rows.splice(rowIndex, 1)
    setSheets(newSheets)
  }

  const addColumn = () => {
    const newSheets = { ...sheets }
    newSheets[activeSheet].columns.push(`Column ${currentSheet.columns.length + 1}`)
    newSheets[activeSheet].rows = newSheets[activeSheet].rows.map((row) => [...row, ""])
    setSheets(newSheets)
  }

  const deleteColumn = (colIndex: number) => {
    const newSheets = { ...sheets }
    newSheets[activeSheet].columns.splice(colIndex, 1)
    newSheets[activeSheet].rows = newSheets[activeSheet].rows.map((row) => {
      const newRow = [...row]
      newRow.splice(colIndex, 1)
      return newRow
    })
    setSheets(newSheets)
  }

  const updateCell = (rowIndex: number, colIndex: number, value: string) => {
    const newSheets = { ...sheets }
    newSheets[activeSheet].rows[rowIndex][colIndex] = value
    setSheets(newSheets)
  }

  const toggleRowSelection = (rowIndex: number) => {
    if (selectedRows.includes(rowIndex)) {
      setSelectedRows(selectedRows.filter((i) => i !== rowIndex))
    } else {
      setSelectedRows([...selectedRows, rowIndex])
    }
  }

  const addAIColumn = () => {
    if (!aiColumnName || !aiPrompt) return

    const newSheets = { ...sheets }
    newSheets[activeSheet].columns.push(aiColumnName)
    newSheets[activeSheet].aiColumns[aiColumnName] = true

    // Add empty cell for each row
    newSheets[activeSheet].rows = newSheets[activeSheet].rows.map((row) => [...row, ""])

    setSheets(newSheets)
    setShowAIDialog(false)

    // Save the last used prompt
    setLastAiPrompt(aiPrompt)

    // Process AI enrichment for selected rows or all rows if none selected
    const rowsToProcess = selectedRows.length > 0 ? selectedRows : currentSheet.rows.map((_, i) => i)
    processAIEnrichment(rowsToProcess, newSheets[activeSheet].columns.length - 1, aiPrompt)
  }

  const processAIEnrichment = async (rowIndices: number[], colIndex: number, promptOverride?: string) => {
    const prompt = promptOverride || lastAiPrompt
    if (!prompt) return

    setLoadingRows((prev) => [...prev, ...rowIndices])

    for (const rowIndex of rowIndices) {
      // Build rowData object with column names as keys
      const rowData: Record<string, string> = {};
      currentSheet.columns.forEach((col: string, idx: number) => {
        rowData[col] = currentSheet.rows[rowIndex][idx];
      });
    
      try {
        const response = await fetch("http://localhost:8000/api/enrich", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            rowData,
            prompt,
            columnName: currentSheet.columns[colIndex]
          }),
        });
    
        if (!response.ok) {
          throw new Error("AI enrichment failed");
        }
    
        const data = await response.json();
    
        // Update the cell with the AI result
        const newSheets = { ...sheets };
        newSheets[activeSheet].rows[rowIndex][colIndex] = data.result;
        setSheets(newSheets);
      } catch (error) {
        // Optionally show a toast or error message
        console.error("AI enrichment error:", error);
      }
    }

    setLoadingRows((prev) => prev.filter((row) => !rowIndices.includes(row)))
    setSelectedRows([])
  }

  const createNewSheet = (name: string) => {
    if (!name || sheets[name]) return

    const newSheets = { ...sheets }
    newSheets[name] = {
      columns: ["Column 1", "Column 2", "Column 3"],
      rows: [["", "", ""]],
      aiColumns: {},
    }

    setSheets(newSheets)
    setActiveSheet(name)
  }

  const exportToCSV = () => {
    const sheet = sheets[activeSheet];
    if (!sheet) return;

    // Helper to escape CSV values
    const escapeCSV = (value: any) => {
      if (value == null) return "";
      // Convert to string, replace newlines with spaces, escape quotes
      return `"${String(value).replace(/\r?\n|\r/g, " ").replace(/"/g, '""')}"`;
    };

    // Build CSV content
    let csv = sheet.columns.map(escapeCSV).join(",") + "\n";
    sheet.rows.forEach((row) => {
      csv += row.map(escapeCSV).join(",") + "\n";
    });

    // Create a blob and trigger download
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${activeSheet.replace(/\s+/g, "_")}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Function to save the current sheet data to the database
  const saveToDatabase = async () => {
    setDbSaveStatus("Saving to database...")
    
    try {
      // Convert sheet data format to array of objects for the backend
      const tableData = currentSheet.rows.map((row: string[]) => {
        const rowObj: Record<string, string> = {}
        currentSheet.columns.forEach((col: string, idx: number) => {
          rowObj[col] = row[idx]
        })
        return rowObj
      })
      
      // Send data to database endpoint
      const response = await fetch("http://localhost:8000/api/table/save-to-database", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ tableData }),
      })
      
      if (!response.ok) {
        throw new Error("Failed to save to database")
      }
      
      const data = await response.json()
      setDbSaveStatus("Saved to database!")
      console.log("Data saved to database:", data.message)
      
      // Clear status after a few seconds
      setTimeout(() => {
        setDbSaveStatus(null)
      }, 3000)
    } catch (error: unknown) {
      console.error("Database save error:", error)
      setDbSaveStatus(`Save failed: ${error instanceof Error ? error.message : "Unknown error"}`)
      
      // Clear error after a few seconds
      setTimeout(() => {
        setDbSaveStatus(null)
      }, 5000)
    }
  }

  const loadFromDatabase = async () => {
    setDbSaveStatus("Loading from database...")
    
    try {
      const response = await fetch("http://localhost:8000/api/table/load-from-database")
      
      if (!response.ok) {
        throw new Error("Failed to load from database")
      }
      
      const data = await response.json()
      
      if (data.tableData && data.tableData.length > 0) {
        // Get columns from metadata to preserve order
        const columns = data.metadata.columns || Object.keys(data.tableData[0])
        
        // Convert array of objects to sheet format
        const rows = data.tableData.map((row: Record<string, string>) => {
          // Use columns array to maintain the same order
          return columns.map((col: string) => row[col] || "")
        })
        
        // Update only the active sheet with database data
        setSheets(prev => ({
          ...prev,
          [activeSheet]: {
            columns,
            rows,
            aiColumns: prev[activeSheet]?.aiColumns || {}
          }
        }))
        
        setDbSaveStatus("Loaded from database!")
      } else {
        setDbSaveStatus("No data found in database")
      }
      
      // Clear status after a few seconds
      setTimeout(() => {
        setDbSaveStatus(null)
      }, 3000)
    } catch (error: unknown) {
      console.error("Database load error:", error)
      setDbSaveStatus(`Load failed: ${error instanceof Error ? error.message : "Unknown error"}`)
      
      // Clear error after a few seconds
      setTimeout(() => {
        setDbSaveStatus(null)
      }, 5000)
    }
  }

  return (
    <div className="flex h-screen flex-col">
      <header className="flex h-14 items-center border-b px-4 lg:h-16">
        <div className="flex items-center gap-2">
          <div className="h-8 w-8 rounded-md bg-blue-600"></div>
          <span className="text-xl font-bold">SmartGrid AI</span>
        </div>
        <div className="ml-auto flex items-center gap-2">
          {dbSaveStatus && (
            <div className={`text-sm flex items-center ${dbSaveStatus.includes('failed') || dbSaveStatus.includes('No data') ? 'text-amber-600' : dbSaveStatus.includes('Saving') ? 'text-blue-600' : 'text-green-600'}`}>
              {dbSaveStatus.includes('Saving') || dbSaveStatus.includes('Loading') ? (
                <Loader2 className="h-4 w-4 mr-1 animate-spin" />
              ) : null}
              {dbSaveStatus}
            </div>
          )}
          <Button 
            variant="outline" 
            size="sm" 
            className="gap-1"
            onClick={saveToDatabase}
          >
            <Database className="h-4 w-4" />
            Save to DB
          </Button>
          <Button 
            variant="outline" 
            size="sm" 
            className="gap-1"
            onClick={loadFromDatabase}
          >
            <Database className="h-4 w-4" />
            Load from DB
          </Button>
          <Button variant="outline" size="sm" className="gap-1">
            <Save className="h-4 w-4" />
            Save
          </Button>
          <Button variant="outline" size="sm" className="gap-1" onClick={exportToCSV}>
            <Download className="h-4 w-4" />
            Export
          </Button>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon" className="rounded-full">
                <User className="h-5 w-5" />
                <span className="sr-only">User menu</span>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuLabel>My Account</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem>
                <Settings className="mr-2 h-4 w-4" />
                Settings
              </DropdownMenuItem>
              <DropdownMenuItem>
                <LogOut className="mr-2 h-4 w-4" />
                Log out
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </header>
      <div className="flex flex-1 overflow-hidden">
        <Sidebar
          activeSheet={activeSheet}
          sheets={Object.keys(sheets)}
          onSelectSheet={setActiveSheet}
          onCreateSheet={(name) => createNewSheet(name)}
        />
        <main className="flex-1 overflow-hidden">
          <div className="flex h-12 items-center justify-between border-b px-4">
            <h1 className="text-lg font-medium">{activeSheet}</h1>
            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" onClick={addColumn} className="gap-1">
                <Plus className="h-4 w-4" />
                Add Column
              </Button>
              <Button
                size="sm"
                className="gap-1"
                onClick={() => {
                  setAiPrompt("")
                  setAiColumnName("")
                  setShowAIDialog(true)
                }}
              >
                <PlusCircle className="h-4 w-4" />
                Add AI Column
              </Button>
            </div>
          </div>
          <div className="overflow-auto p-4">
            <div className="rounded-md border">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b bg-muted/50">
                      <th className="w-10 p-2">
                        <div className="flex justify-center">
                          <Button variant="ghost" size="icon" className="h-6 w-6" onClick={addRow}>
                            <Plus className="h-4 w-4" />
                            <span className="sr-only">Add row</span>
                          </Button>
                        </div>
                      </th>
                      {currentSheet.columns.map((column, colIndex) => (
                        <th key={colIndex} className="border-l p-2 text-left font-medium">
                          <div className="flex items-center gap-2">
                            <span className={currentSheet.aiColumns[column] ? "text-blue-600" : ""}>
                              {column} {currentSheet.aiColumns[column] && "(AI)"}
                            </span>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="ml-auto h-6 w-6"
                              onClick={() => deleteColumn(colIndex)}
                            >
                              <Trash2 className="h-3 w-3" />
                              <span className="sr-only">Delete column</span>
                            </Button>
                          </div>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {currentSheet.rows.map((row, rowIndex) => (
                      <tr key={rowIndex} className="group border-b">
                        <td className="w-10 p-2">
                          <div className="flex items-center justify-center gap-1">
                            <input
                              type="checkbox"
                              className="h-4 w-4 rounded border-gray-300"
                              checked={selectedRows.includes(rowIndex)}
                              onChange={() => toggleRowSelection(rowIndex)}
                            />
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-6 w-6 opacity-0 group-hover:opacity-100"
                              onClick={() => deleteRow(rowIndex)}
                            >
                              <Trash2 className="h-3 w-3" />
                              <span className="sr-only">Delete row</span>
                            </Button>
                          </div>
                        </td>
                        {row.map((cell, colIndex) => (
                          <td key={colIndex} className="border-l p-0">
                            <div className="relative">
                              {loadingRows.includes(rowIndex) &&
                              currentSheet.aiColumns[currentSheet.columns[colIndex]] ? (
                                <div className="flex h-10 items-center justify-center p-2">
                                  <Loader2 className="h-4 w-4 animate-spin text-blue-600" />
                                </div>
                              ) : (
                                <div className="relative group/cell">
                                  <Input
                                    value={cell}
                                    onChange={(e) => updateCell(rowIndex, colIndex, e.target.value)}
                                    className={`h-10 border-0 rounded-none focus:ring-1 focus:ring-inset ${
                                      currentSheet.aiColumns[currentSheet.columns[colIndex]]
                                        ? "bg-blue-50 text-blue-600 pr-8"
                                        : ""
                                    }`}
                                  />
                                  {currentSheet.aiColumns[currentSheet.columns[colIndex]] && (
                                    <Button
                                      variant="ghost"
                                      size="icon"
                                      className="absolute right-1 top-1/2 h-6 w-6 -translate-y-1/2 opacity-0 group-hover/cell:opacity-100"
                                      onClick={() => processAIEnrichment([rowIndex], colIndex, aiPrompt)}
                                      title="Regenerate"
                                    >
                                      <RefreshCw className="h-3 w-3" />
                                      <span className="sr-only">Regenerate</span>
                                    </Button>
                                  )}
                                </div>
                              )}
                            </div>
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </main>
      </div>

      {/* AI Column Dialog */}
      <Dialog open={showAIDialog} onOpenChange={setShowAIDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Add AI Column</DialogTitle>
            <DialogDescription>Create a new column with AI-generated insights based on your data.</DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <Label htmlFor="column-name">Column Name</Label>
              <Input
                id="column-name"
                placeholder="e.g., Classification, Sentiment, Summary"
                value={aiColumnName}
                onChange={(e) => setAiColumnName(e.target.value)}
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="ai-prompt">AI Prompt</Label>
              <Input
                id="ai-prompt"
                placeholder="e.g., Classify this major as Engineer or Non-Engineer"
                value={aiPrompt}
                onChange={(e) => setAiPrompt(e.target.value)}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowAIDialog(false)}>
              Cancel
            </Button>
            <Button onClick={addAIColumn}>Create AI Column</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
