import config
import os

# ---------- utils ----------

def ask(msg):
    return input(f"{msg} [y/N]: ").strip().lower() == "y"


def run(module_name):
    print(f"\n▶ Ejecutando: {module_name}")
    mod = __import__(module_name, fromlist=["run"])
    mod.run()


def delete_if_exists(path):
    if os.path.exists(path):
        os.remove(path)
        print(f"🗑 Borrado: {path}")
    else:
        print(f"· No existe: {path}")


def nuke_everything():
    print("\nBorrando datos y modelos anteriores")

    files_to_delete = [
        "data/eth_ohlcv_1h.csv",
        "data/eth_ohlcv_30m.csv",
        "features/eth_features_1h.csv",
        "ml/feature_cols_regime.pkl",
        "ml/model_regime.pkl",
        "rl/dqn_trader_regime.zip",
    ]

    for f in files_to_delete:
        delete_if_exists(f)

    print("Estado limpio.\n")


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
8 - 🔁 GRID REGIME_THRESHOLD (0.64 → 0.80)
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

        elif choice == "1":
            if ask("Esto actualiza/mergea la data. ¿Seguro?"):
                run("data.download_data")

        elif choice == "2":
            if ask("Recalcular features desde cero?"):
                run("features.build_features")

        elif choice == "3":
            run("ml.train_ml")

        elif choice == "4":
            if ask("Entrenar modelo RL ahora? (tarda)"):
                run("rl.train_rl")

        elif choice == "5":
            run("rl.backtest")

        elif choice == "6":
            if ask("LIVE/PAPER puede perder guita. ¿Seguimos?"):
                run("rl.live")
                
        elif choice == "8":
            print("\n⚠️  GRID SEARCH HARDCODEADO DE REGIME_THRESHOLD")
            print("Esto borra TODO y corre el pipeline completo por cada valor.")

            if not ask("¿Seguro que querés hacer esto?"):
                continue

            for rt in [round(x, 2) for x in [0.660 + i * 0.02 for i in range(9)]]:
                print("\n" + "=" * 50)
                print(f"▶▶ REGIME_THRESHOLD = {rt}")
                print("=" * 50)

                # set config
                config.REGIME_THRESHOLD = rt

                # borrar todo
                nuke_everything()

                # pipeline completo
                run("data.download_data")
                run("features.build_features")
                run("ml.train_ml")
                run("rl.train_rl")
                run("rl.backtest")

            print("\n✅ GRID SEARCH TERMINADO")


        elif choice == "9":
            nuke_everything()

            run("data.download_data")
            run("features.build_features")
            run("ml.train_ml")
            run("rl.train_rl")
            run("rl.backtest")

        else:
            print("Opción inválida.")


if __name__ == "__main__":
    main()
