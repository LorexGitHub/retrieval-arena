export function HeroBackground() {
  return (
    <div className="fixed top-0 left-0 w-full h-screen z-[-1] flex flex-col items-center justify-start pointer-events-none bg-bg pt-24">
      <div className="relative w-[140px] h-[140px] opacity-40">
        <div className="absolute inset-0 rounded-full bg-surface/50 border border-border/50" />
        <div className="absolute inset-[15px] rounded-full bg-accent/10 border border-accent/20 flex items-center justify-center">
          <div className="w-3 h-3 rounded-[3px] bg-accent/60 rotate-45" />
        </div>
      </div>
    </div>
  )
}
