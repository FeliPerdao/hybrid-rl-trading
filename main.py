import config
import os

# ---------- utils ----------

def ask(msg):
    return input(f"{msg} [y/N]: ").strip().lower() == "y"


def run(module_name, *args, **kwargs):
    print(f"\n▶ Ejecutando: {module_name}")
    mod = __import__(module_name, fromlist=["run", "main"])
    if hasattr(mod, "main"):
        return mod.main(*args, **kwargs)
    else:
        mod.run()


def delete_if_exists(path):
    if os.path.exists(path):
        os.remove(path)
        print(f"🗑 Borrado: {path}")
    else:
        print(f"· No existe: {path}")


# ---------- main ----------

def main():
    print("""
==============================
TRADING SYSTEM CONTROL
==============================
1 - Actualizar DATA
2 - Recalcular FEATURES
3 - Entrenar MODELO ML
4 - Entrenar MODELO RL
5 - BACKTEST
6 - LIVE / PAPER
8 - 🔁 GRID REGIME_THRESHOLD
9 - 🚀 RUN ALL FROM ZERO
0 - Salir
""")

    while True:
        try:
            choice = input("Elegí opción: ").strip()
        except KeyboardInterrupt:
            print("\nChau.")
            return

        if choice == "0":
            print("Salir.")
            return

        elif choice == "3":
            import ml.train_ml as train_ml

            print("\nSeleccioná el timeframe del modelo ML\n")

            for i, tf in enumerate(train_ml.AVAILABLE_TFS, 1):
                print(f"{i} - {tf}")

            try:
                sel = int(input("\nElegí UNO: ").strip())
                tf = train_ml.AVAILABLE_TFS[sel - 1]
            except:
                print("❌ Selección inválida.")
                continue

            print(f"\n🧠 Entrenando modelo ML para timeframe: {tf}")
            train_ml.main(timeframes=[tf])

        elif choice == "4":
            if ask("Entrenar modelo RL ahora? (tarda)"):
                run("rl.train_rl")

        elif choice == "5":
            run("rl.backtest")

        elif choice == "6":
            if ask("LIVE/PAPER puede perder guita. ¿Seguimos?"):
                run("rl.live")

        else:
            print("Opción inválida.")


if __name__ == "__main__":
    main()
