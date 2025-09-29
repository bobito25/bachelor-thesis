

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a e)
(ontable b)
(ontable c)
(on d b)
(on e d)
(on f a)
(clear c)
(clear f)
)
(:goal
(and
(on a c)
(on b a)
(on d e)
(on e b)
(on f d))
)
)


