export function StatusMessage({ children, error = false }: { children: React.ReactNode; error?: boolean }) {
  return <div className={`status-message ${error ? 'is-error' : ''}`} role={error ? 'alert' : 'status'}>{children}</div>;
}
