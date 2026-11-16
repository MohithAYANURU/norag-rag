/**
 * Workstream 03: Evidence Export Module
 * Reads saved evidence entries from localStorage and exports them as a formatted Markdown file.
 * Shared entry format: { id, document_id, source, chunk_id, text, note }
 */

const STORAGE_KEY = 'rag_evidence_notebook';

function getStoredEvidence() {
    try {
        const data = localStorage.getItem(STORAGE_KEY);
        return data ? JSON.parse(data) : [];
    } catch (e) {
        console.error("Failed to load evidence from localStorage", e);
        return [];
    }
}

function exportEvidenceToMarkdown() {
    const entries = getStoredEvidence();
    
    if (entries.length === 0) {
        alert("No saved evidence found in the notebook to export.");
        return;
    }

    let markdownContent = `# RAG Evidence Notebook Export\n`;
    markdownContent += `Exported on: ${new Date().toLocaleString()}\n`;
    markdownContent += `Total Saved Passages: ${entries.length}\n\n`;
    markdownContent += `---\n\n`;

    entries.forEach((entry, index) => {
        markdownContent += `## Citation ${index + 1}\n\n`;
        markdownContent += `- **Source:** \`${entry.source || 'Unknown Source'}\`\n`;
        markdownContent += `- **Chunk ID:** \`${entry.chunk_id || 'N/A'}\`\n`;
        markdownContent += `- **Document ID:** \`${entry.document_id || 'N/A'}\`\n\n`;
        
        markdownContent += `### Excerpt\n> ${(entry.text || '').replace(/\n/g, '\n> ')}\n\n`;
        
        if (entry.note && entry.note.trim() !== '') {
            markdownContent += `### Notes\n`;
            entry.note.split('\n').forEach(line => {
                markdownContent += `${line}\n`;
            });
            markdownContent += `\n`;
        }
        
        markdownContent += `---\n\n`;
    });

    const blob = new Blob([markdownContent], { type: 'text/markdown;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    
    link.href = url;
    link.download = `rag-evidence-export-${new Date().toISOString().slice(0, 10)}.md`;
    document.body.appendChild(link);
    link.click();
    
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

window.exportEvidenceToMarkdown = exportEvidenceToMarkdown;